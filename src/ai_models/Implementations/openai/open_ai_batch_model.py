from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Optional
import math


class OpenAIBatchModel(BaseBatchModel):
    def __init__(
        self,
        model_name,
        tokenizer,
        input_cost_per_million,
        output_cost_per_million,
        batch_queue_limit,
    ):
        super().__init__(
            model_name,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            batch_queue_limit,
        )
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        }
        self.requests.append(request)

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> List[str]:
        batches = self._create_batch(metadata, test_session_id, benchmark_name)

        return batches

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        batch_status = self.client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.status}")

        if batch_status.status == "completed":
            output_file_path = self._download_results(
                batch_status.output_file_id, benchmark_name, test_session_id, batch_id
            )
            return self.process_batch_results(output_file_path)

        return None

    def cancel_batch(self, batch_id: str):
        return self.client.batches.cancel(batch_id)

    def list_batches(self, limit: int = 100):
        return self.client.batches.list(limit=limit)

    def process_batch_results(self, output_file_path: str) -> BatchResponse:
        results = []
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data["custom_id"]
                response_data = data["response"]

                if response_data["status_code"] == 200:
                    response = response_data["body"]["choices"][0]["message"]["content"]
                    usage_data = response_data["body"]["usage"]
                    usage = Usage(
                        usage_data["prompt_tokens"], usage_data["completion_tokens"]
                    )
                    status = "success"
                else:
                    response = None
                    usage = None
                    status = "failed"

                results.append(
                    BatchResponseItem(
                        custom_id=custom_id,
                        response=response,
                        usage=usage,
                        status=status,
                    )
                )

        return BatchResponse(results)

    def _create_batch(self, metadata=None, test_session_id=None, benchmark_name=None):
        total_tokens = sum(
            self.estimate_tokens_amount(req["body"]["messages"])
            for req in self.requests
        )
        tokens_per_batch = int(
            self.batch_queue_limit * 0.9
        )  # 90% of the batch queue limit because im not sure yet how tokenizer handles this imputs it needs more work
        num_batches = math.ceil(total_tokens / tokens_per_batch)

        batch_ids = []

        if num_batches == 1:
            id = self.create_single_batch(
                self.requests, test_session_id, benchmark_name, metadata
            )
            batch_ids.append(id)
        else:
            current_batch = []
            current_tokens = 0
            for request in self.requests:
                request_tokens = self.estimate_tokens_amount(
                    request["body"]["messages"]
                )
                if current_tokens + request_tokens > tokens_per_batch:
                    id = self.create_single_batch(
                        current_batch, test_session_id, benchmark_name, metadata
                    )
                    batch_ids.append(id)
                    current_batch = [request]
                    current_tokens = request_tokens
                else:
                    current_batch.append(request)
                    current_tokens += request_tokens

            if current_batch:
                id = self.create_single_batch(
                    current_batch, test_session_id, benchmark_name, metadata
                )
                batch_ids.append(id)

        print(f"Created {len(batch_ids)} batches")
        return batch_ids

    def create_single_batch(
        self, requests, test_session_id=None, benchmark_name=None, metadata=None
    ):
        batch_file = self._create_batch_file(requests, test_session_id, benchmark_name)
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )
        return batch.id

    def _create_batch_file(self, requests, test_session_id, benchmark_name):
        input_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_requests_{os.urandom(8).hex()}.jsonl"
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        with open(input_file_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="batch")
        return uploaded_file

    def _download_results(self, file_id, benchmark_name, test_session_id, batch_id):
        output_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_{batch_id}_results.jsonl"
        if os.path.exists(output_file_path):
            return output_file_path

        file_content = self.client.files.content(file_id)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(file_content.text)
        return output_file_path

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        # Convert messages list to string format that can be processed by base class method
        messages_str = ""
        for message in messages:
            messages_str += f"{message['role']}: {message['content']}\n"
        return super().estimate_tokens_amount(messages_str)

    def get_input_file_url(self, batch_id: str) -> Optional[str]:
        try:
            batch_status = self.client.batches.retrieve(batch_id)
            return batch_status.input_file_id
        except Exception as e:
            print(f"Error retrieving input file URL for batch {batch_id}: {e}")
            return None

    def retry_batch(
        self,
        batch_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Retry failed batch using its input file"""
        try:
            input_file_id = self.get_input_file_url(batch_id)
            if not input_file_id:
                print(f"Could not retrieve input file for batch {batch_id}")
                return None

            # Create new batch using the same input file
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
            print(f"Created new batch {batch.id} from failed batch {batch_id}")
            return batch.id

        except Exception as e:
            print(f"Error retrying batch {batch_id}: {e}")
            return None
