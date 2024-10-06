from ...interfaces import BatchRunnerInterface
import os
import json
from threading import Lock


class OpenAIBatchRunner(BatchRunnerInterface):
    def __init__(self, client):
        self.client = client
        self.lock = Lock()
        self.requests = []

    def add_request(self, custom_id, model, messages, max_tokens=1):
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": messages, "max_tokens": max_tokens},
        }
        self.requests.append(request)

    def run_batch(self, benchmark_name, model_name, metadata=None):
        input_file_path = f"batch/{benchmark_name}/{model_name}_batch_requests.jsonl"
        self._save_to_file(input_file_path)
        input_file_id = self._upload_file(input_file_path)
        batch = self._create_batch(input_file_id, metadata=metadata)
        return batch.id

    def check_for_results(self, benchmark_name, model_name, batch_id):
        status = self._check_status(batch_id)
        print(f"Batch status: {status.status}")

        if status.status == "completed":
            file_id = status.output_file_id
            results = self.client.files.content(file_id)

            output_file_path = (
                f"batch/{benchmark_name}/{model_name}_batch_results.jsonl"
            )
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(results.text)
            return output_file_path

        return None

    def _save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            for request in self.requests:
                f.write(json.dumps(request) + "\n")

    def _upload_file(self, file_path):
        with open(file_path, "rb") as file:
            response = self.client.files.create(file=file, purpose="batch")
        return response.id

    def _create_batch(self, input_file_id, metadata=None):
        with self.lock:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
        return batch

    def _check_status(self, batch_id):
        with self.lock:
            return self.client.batches.retrieve(batch_id)
