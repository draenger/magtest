import json
from typing import List
from .implementations.openai import OpenAIBatchManager
from .implementations.anthropic import AnthropicBatchManager
from .implementations.google import GoogleBatchManager
from colorama import init
from collections import defaultdict


class BatchProgressManager:
    def __init__(self, batch_job_repo):
        init()
        self.openai_manager = OpenAIBatchManager()
        self.anthropic_manager = AnthropicBatchManager()
        self.google_manager = GoogleBatchManager()
        self.config_file = "model_config.json"
        self.model_config = self._load_model_config()
        self.batch_job_repo = batch_job_repo

    def show_batch_progress(self, batch_info: List[dict]):
        if not batch_info:
            print("No batch information provided")
            return

        for item in batch_info:
            try:
                model_name = item.get("model_name", "unknown")
                batch_id = item.get("batch_id")
                if not batch_id:
                    print(f"No batch ID provided for model {model_name}")
                    continue

                provider = self._get_model_provider(model_name)
                if provider == "openai":
                    self.openai_manager.show_batch_progress([batch_id])
                elif provider == "anthropic":
                    self.anthropic_manager.show_batch_progress([batch_id])
                elif provider == "google":
                    self.google_manager.show_batch_progress([batch_id])
                else:
                    print(f"Unsupported model provider: {provider}")
            except Exception as e:
                print(f"Error processing batch info: {e}")
                continue

    def show_batch_progress_from_db(self, test_session_id: int):
        try:
            batch_jobs = self.batch_job_repo.get_by_test_session(test_session_id)
            if not batch_jobs:
                print(f"No batch jobs found for test session {test_session_id}")
                return

            # Group jobs by benchmark, provider and model
            provider_groups = defaultdict(lambda: defaultdict(list))
            for job in batch_jobs:
                benchmark_name = getattr(job, "benchmark_name", "unknown")
                model_name = getattr(job, "model_name", "unknown")
                provider = self._get_model_provider(model_name)
                provider_groups[benchmark_name][provider].append(job)

            # Show progress for each benchmark, grouped by provider
            for benchmark_name, provider_jobs in provider_groups.items():
                print(f"\nBenchmark: {benchmark_name}")
                print("=" * 60)

                for provider, jobs in provider_jobs.items():
                    print(f"\n{provider.upper()} Models:")
                    print("-" * 30)

                    # Group jobs by model within provider
                    model_jobs = defaultdict(list)
                    for job in jobs:
                        model_jobs[job.model_name].append(job)

                    # Show progress for each model
                    for model_name, model_batch_jobs in model_jobs.items():
                        print(f"\nModel: {model_name}")
                        for job in model_batch_jobs:
                            try:
                                batch_id = getattr(job, "batch_id", None)
                                if not batch_id:
                                    print(f"No batch ID found for job")
                                    continue

                                if provider == "openai":
                                    self.openai_manager.show_batch_progress([batch_id])
                                elif provider == "anthropic":
                                    self.anthropic_manager.show_batch_progress(
                                        [batch_id]
                                    )
                                elif provider == "google":
                                    self.google_manager.show_batch_progress([batch_id])
                                else:
                                    print(f"Unsupported model provider: {provider}")
                            except Exception as e:
                                print(
                                    f"Error showing progress for batch {getattr(job, 'batch_id', 'unknown')}: {e}"
                                )
                                continue

        except Exception as e:
            print(f"Error retrieving batch jobs: {e}")

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
