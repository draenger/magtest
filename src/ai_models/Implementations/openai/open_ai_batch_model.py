from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Optional


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
    ) -> str:
        input_file_path = self._prepare_batch(benchmark_name, test_session_id)
        input_file = self._upload_file(input_file_path)
        batch = self._create_batch(input_file.id, metadata)
        print(f"Batch created: {batch}")
        return batch.id

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        batch_status = self.client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.status}")

        if batch_status.status == "completed":
            output_file_path = self._download_results(
                batch_status.output_file_id, benchmark_name, test_session_id
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

    def _prepare_batch(self, benchmark_name, test_session_id):
        input_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_requests.jsonl"
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        with open(input_file_path, "w") as f:
            for request in self.requests:
                f.write(json.dumps(request) + "\n")
        return input_file_path

    def _upload_file(self, file_path):
        with open(file_path, "rb") as file:
            return self.client.files.create(file=file, purpose="batch")

    def _create_batch(self, input_file_id, metadata=None):
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )

    def _download_results(self, file_id, benchmark_name, test_session_id):
        output_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_results.jsonl"
        if os.path.exists(output_file_path):
            return output_file_path

        file_content = self.client.files.content(file_id)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(file_content.text)
        return output_file_path

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        return super().estimate_tokens_amount(messages)
