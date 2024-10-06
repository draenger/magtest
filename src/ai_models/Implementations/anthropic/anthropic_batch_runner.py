import os
import json
from ...interfaces import BatchRunnerInterface


class AnthropicBatchRunner(BatchRunnerInterface):
    def __init__(self, client):
        self.client = client
        self.requests = []

    def add_request(self, custom_id, model, messages, max_tokens=1):
        request = {
            "custom_id": custom_id,
            "model": model,
            "prompt": messages[-1]["content"],
            "max_tokens_to_sample": max_tokens,
        }
        self.requests.append(request)

    def run_batch(self, benchmark_name, model_name, metadata=None):
        # Implementation of batch running (similar to OpenAIBatchRunner)
        input_file_path = f"batch/{benchmark_name}/{model_name}_batch_requests.json"
        self._save_to_file(input_file_path)
        # Here you would implement the actual batch running logic for Anthropic
        # This is a placeholder and should be replaced with actual Anthropic batch API calls
        batch_id = f"anthropic_batch_{benchmark_name}_{model_name}"
        return batch_id

    def check_for_results(self, benchmark_name, model_name, batch_id):
        # Implementation of checking for batch results (similar to OpenAIBatchRunner)
        # This is a placeholder and should be replaced with actual Anthropic batch result checking
        output_file_path = f"batch/{benchmark_name}/{model_name}_batch_results.json"
        if os.path.exists(output_file_path):
            return output_file_path
        return None

    def _save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.requests, f)
