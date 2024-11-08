from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List


class OpenAIBatchManager:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_batch_info(self, batch_id):
        try:
            return self.client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"Error retrieving batch info for {batch_id}: {e}")
            return None

    def display_batch_progress(self, batch_info):
        if not batch_info:
            print(f"No batch info available")
            return

        try:
            total = getattr(batch_info.request_counts, "total", 0)
            completed = getattr(batch_info.request_counts, "completed", 0)
            failed = getattr(batch_info.request_counts, "failed", 0)
            status = getattr(batch_info, "status", "unknown")

            # If batch is completed successfully (all tests completed, none failed)
            if status == "completed" and completed == total and failed == 0:
                print(f"Batch ID: {batch_info.id} - fully completed")
                print("-" * 60)
                return

            # If batch has failed
            if status == "failed" or failed == total:
                print(f"Batch ID: {batch_info.id} - failed")
                print("-" * 60)
                return

            if total == 0:
                print(f"Batch ID: {batch_info.id}")
                print(f"Status: {status}")
                print("No request count data available")
                print("-" * 60)
                return

            in_progress = total - completed - failed

            print(f"Batch ID: {batch_info.id}")
            print(f"Status: {status}")
            print(f"Total requests: {total}")

            bar_length = 50
            completed_length = int((completed / total) * bar_length) if total > 0 else 0
            failed_length = int((failed / total) * bar_length) if total > 0 else 0
            in_progress_length = bar_length - completed_length - failed_length

            progress_bar = (
                "█" * completed_length + "x" * failed_length + " " * in_progress_length
            )

            print(f"Progress: [{progress_bar}]")
            print(
                f"Completed: {completed} | Failed: {failed} | In Progress: {in_progress}"
            )

            if total > 0:
                completed_percent = (completed / total) * 100
                failed_percent = (failed / total) * 100
                in_progress_percent = (in_progress / total) * 100

                print(
                    f"Completed: {completed_percent:.2f}% | Failed: {failed_percent:.2f}% | In Progress: {in_progress_percent:.2f}%"
                )
            print("-" * 60)
        except Exception as e:
            print(f"Error displaying batch progress: {e}")
            print(f"Batch ID: {getattr(batch_info, 'id', 'unknown')}")
            print(f"Status: {getattr(batch_info, 'status', 'unknown')}")
            print("Unable to display detailed progress information")
            print("-" * 60)

    def show_batch_progress(self, batch_ids: List[str]):
        for batch_id in batch_ids:
            batch_info = self.get_batch_info(batch_id)
            if batch_info:
                self.display_batch_progress(batch_info)
            else:
                print(f"No information available for batch {batch_id}")
                print("-" * 60)
