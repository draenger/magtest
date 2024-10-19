from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
from anthropic import Anthropic
import os
import json
from typing import List, Optional


class AnthropicBatchModel(BaseBatchModel):
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
        self.anthropic_client = Anthropic()

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = {
            "custom_id": custom_id,
            "model": self.model_name,
            "prompt": messages[-1]["content"],
            "max_tokens_to_sample": max_tokens,
        }
        self.requests.append(request)

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        input_file_path = self._prepare_batch(benchmark_name, test_session_id)
        # Here you would implement the actual batch running logic for Anthropic
        # This is a placeholder and should be replaced with actual Anthropic batch API calls
        batch_id = f"anthropic_batch_{benchmark_name}_{self.model_name}"
        return batch_id

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        # Implementation of checking for batch results
        # This is a placeholder and should be replaced with actual Anthropic batch result checking
        output_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_results.jsonl"
        if os.path.exists(output_file_path):
            return self.process_batch_results(output_file_path)
        return None

    def cancel_batch(self, batch_id: str):
        # Implement batch cancellation for Anthropic
        return {"status": "cancelled", "batch_id": batch_id}

    def list_batches(self, limit: int = 10):
        # Implement batch listing for Anthropic
        return []

    def process_batch_results(self, output_file_path: str) -> BatchResponse:
        results = []
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data["custom_id"]
                response_data = data["response"]

                if response_data["status"] == "success":
                    response = response_data["completion"]
                    usage_data = response_data["usage"]
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

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        total_tokens = 0
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                total_tokens += super().estimate_tokens_amount(message["content"])
            else:
                print(f"Warning: Unexpected message format: {message}")
        return total_tokens
