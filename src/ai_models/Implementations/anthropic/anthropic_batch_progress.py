from anthropic import Anthropic
import os
from dotenv import load_dotenv
from typing import List


class AnthropicBatchManager:
    def __init__(self):
        load_dotenv()
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_batch_info(self, batch_id):
        try:
            return self.client.beta.messages.batches.retrieve(batch_id)
        except Exception as e:
            print(f"Error retrieving batch info for {batch_id}: {e}")
            return None

    def display_batch_progress(self, batch_info):
        if not batch_info:
            return

        total = sum(batch_info.request_counts.__dict__.values())
        completed = batch_info.request_counts.succeeded
        failed = batch_info.request_counts.errored + batch_info.request_counts.expired
        in_progress = batch_info.request_counts.processing

        print(f"Batch ID: {batch_info.id}")
        print(f"Status: {batch_info.processing_status}")
        print(f"Total requests: {total}")

        bar_length = 50
        completed_length = int(completed / total * bar_length)
        failed_length = int(failed / total * bar_length)
        in_progress_length = int(in_progress / total * bar_length)

        progress_bar = (
            "█" * completed_length + "x" * failed_length + " " * in_progress_length
        )

        print(f"Progress: [{progress_bar}]")
        print(f"Completed: {completed} | Failed: {failed} | In Progress: {in_progress}")

        completed_percent = (completed / total) * 100
        failed_percent = (failed / total) * 100
        in_progress_percent = (in_progress / total) * 100

        print(
            f"Completed: {completed_percent:.2f}% | Failed: {failed_percent:.2f}% | In Progress: {in_progress_percent:.2f}%"
        )
        print("-" * 60)

    def show_batch_progress(self, batch_ids: List[str]):
        for batch_id in batch_ids:
            batch_info = self.get_batch_info(batch_id)
            if batch_info:
                self.display_batch_progress(batch_info)
