from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
import random
import json
import os
from typing import List, Optional, Union


class TestBatchModel(BaseBatchModel):
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

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = {
            "custom_id": custom_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        self.requests.append(request)

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        batch_id = f"test_batch_{random.randint(1000, 9999)}"
        self._prepare_batch(benchmark_name, test_session_id)
        return batch_id

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        output_file_path = self._generate_results(benchmark_name, test_session_id)
        return self.process_batch_results(output_file_path)

    def cancel_batch(self, batch_id: str):
        return {"status": "cancelled", "batch_id": batch_id}

    def list_batches(self, limit: int = 10):
        return [
            {
                "id": f"test_batch_{i}",
                "status": random.choice(["completed", "in_progress"]),
            }
            for i in range(limit)
        ]

    def process_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        output_file_path = self._generate_results(benchmark_name, test_session_id)
        results = []
        try:
            with open(output_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    custom_id = data["custom_id"]
                    response_data = data["response"]["body"]

                    response = response_data["choices"][0]["message"]["content"]
                    usage_data = response_data["usage"]
                    usage = Usage(
                        usage_data["prompt_tokens"], usage_data["completion_tokens"]
                    )
                    status = "success"

                    results.append(
                        BatchResponseItem(
                            custom_id=custom_id,
                            response=response,
                            usage=usage,
                            status=status,
                        )
                    )

            return BatchResponse(results)
        except Exception as e:
            print(f"Error processing batch results: {str(e)}")
            return None

    def _prepare_batch(self, benchmark_name, test_session_id):
        input_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_requests.jsonl"
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        with open(input_file_path, "w") as f:
            for request in self.requests:
                f.write(json.dumps(request) + "\n")
        return input_file_path

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        total_tokens = 0
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                total_tokens += super().estimate_tokens_amount(message["content"])
            else:
                print(f"Warning: Unexpected message format: {message}")
        return total_tokens

    def _generate_results(self, benchmark_name, test_session_id):
        input_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_requests.jsonl"
        output_file_path = f"src/batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_results.jsonl"

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(input_file_path, "r", encoding="utf-8") as input_file, open(
            output_file_path, "w", encoding="utf-8"
        ) as output_file:

            for line in input_file:
                request = json.loads(line)
                custom_id = request["custom_id"]
                messages = request["messages"]

                prompt_tokens = self.estimate_tokens_amount(messages)

                result = {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": random.choice(["A", "B", "C", "D"])
                                    }
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 1,
                                "total_tokens": prompt_tokens + 1,
                            },
                        },
                    },
                }
                output_file.write(json.dumps(result) + "\n")

        return output_file_path

    def retry_batch(
        self,
        batch_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        print("Retry not implemented for Test Batch Model")
        return None

    def get_input_file_url(self, batch_id: str) -> Optional[str]:
        print("Get input file URL not implemented for Test Batch Model")
        return None

    def check_batch_status(self, batch_id: str) -> Optional[str]:
        return "completed"  # Test model always returns completed
