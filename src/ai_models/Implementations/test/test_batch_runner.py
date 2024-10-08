import random
import json
import os
import time


class TestBatchRunner:
    def __init__(self, client, batch_queue_limit):
        self.client = client
        self.batch_queue_limit = batch_queue_limit
        self.requests = []

    def add_request(self, custom_id, model, messages, max_tokens=1):
        request = {
            "custom_id": custom_id,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        self.requests.append(request)

    def run_batch(self, benchmark_name, model_name, metadata=None):
        batch_id = f"test_batch_{random.randint(1000, 9999)}"
        input_file_path = f"batch/{benchmark_name}/{model_name}_batch_requests.jsonl"
        self._save_to_file(input_file_path)
        return batch_id

    def check_for_results(self, benchmark_name, model_name, batch_id):
        # Simulate batch processing completion
        if random.random() < 0.8:  # 80% chance of completion
            results = [
                {
                    "custom_id": req["custom_id"],
                    "choices": [
                        {"message": {"content": random.choice(["A", "B", "C", "D"])}}
                    ],
                }
                for req in self.requests
            ]
            output_file_path = f"batch/{benchmark_name}/{model_name}_batch_results.json"
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as f:
                json.dump(results, f)
            return output_file_path
        return None

    def _save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            for request in self.requests:
                f.write(json.dumps(request) + "\n")
