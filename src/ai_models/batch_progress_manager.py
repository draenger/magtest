import json
from typing import List
from .implementations.openai import OpenAIBatchManager
from .implementations.anthropic import AnthropicBatchManager
from colorama import init


class BatchProgressManager:
    def __init__(self, batch_job_repo):
        init()
        self.openai_manager = OpenAIBatchManager()
        self.anthropic_manager = AnthropicBatchManager()
        self.config_file = "model_config.json"
        self.model_config = self._load_model_config()
        self.batch_job_repo = batch_job_repo

    def show_batch_progress(self, batch_info: List[dict]):
        for item in batch_info:
            model_name = item["model_name"]
            batch_id = item["batch_id"]
            provider = self._get_model_provider(model_name)
            if provider == "openai":
                self.openai_manager.show_batch_progress([batch_id])
            elif provider == "anthropic":
                self.anthropic_manager.show_batch_progress([batch_id])
            else:
                raise ValueError(f"Unsupported model provider: {provider}")

    def show_batch_progress_from_db(self, test_session_id: int):
        batch_jobs = self.batch_job_repo.get_by_test_session(test_session_id)

        # Group batch jobs by benchmark and model
        grouped_jobs = {}
        for job in batch_jobs:
            key = (job.benchmark_name, job.model_name)
            if key not in grouped_jobs:
                grouped_jobs[key] = []
            grouped_jobs[key].append(job)

        # Show progress for each benchmark and model
        for (benchmark_name, model_name), jobs in grouped_jobs.items():
            print(f"\nBenchmark: {benchmark_name}, Model: {model_name}")
            for job in jobs:
                provider = self._get_model_provider(model_name)
                if provider == "openai":
                    self.openai_manager.show_batch_progress([job.batch_id])
                elif provider == "anthropic":
                    self.anthropic_manager.show_batch_progress([job.batch_id])
                else:
                    print(f"Unsupported model provider: {provider}")

    def _load_model_config(self):
        with open(self.config_file, "r") as f:
            return json.load(f)

    def _get_model_provider(self, model_name: str) -> str:
        for provider, models in self.model_config["models"].items():
            if any(model["model_name"] == model_name for model in models):
                return provider
        raise ValueError(f"Model {model_name} not found in configuration")

    def update_batch_status(self, batch_id: str, status: str):
        self.batch_job_repo.update_status(batch_id, status)

    def get_pending_batches(self):
        return self.batch_job_repo.get_pending_jobs()

    def get_batch_status(self, batch_id: str):
        return self.batch_job_repo.get_job_status(batch_id)
