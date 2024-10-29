from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
from vertexai.preview.batch_prediction import BatchPredictionJob
from dotenv import load_dotenv
import os
import json
from typing import List, Optional
import time


class GoogleBatchModel(BaseBatchModel):
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
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.bucket_name = os.environ.get("GOOGLE_CLOUD_BUCKET")
        vertexai.init(project=self.project_id, location="us-central1")

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = {
            "custom_id": custom_id,
            "instances": [
                {
                    "messages": messages,
                    "max_output_tokens": max_tokens,
                }
            ],
        }
        self.requests.append(request)

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        input_file_path = self._prepare_batch(benchmark_name, test_session_id)
        output_uri = (
            f"gs://{self.bucket_name}/{benchmark_name}/{test_session_id}/output"
        )

        # Submit a batch prediction job with Gemini model
        batch_prediction_job = BatchPredictionJob.submit(
            source_model="publishers/google/models/gemini-1.5-flash-002",
            input_dataset=input_file_path,
            output_uri_prefix=output_uri,
        )

        # Check job status
        print(f"Job resource name: {batch_prediction_job.resource_name}")
        print(f"Model resource name with the job: {batch_prediction_job.model_name}")
        print(f"Job state: {batch_prediction_job.state.name}")

        # Refresh the job until complete
        while not batch_prediction_job.has_ended:
            time.sleep(5)
            batch_prediction_job.refresh()

        # Check if the job succeeds
        if batch_prediction_job.has_succeeded:
            print("Job succeeded!")
        else:
            print(f"Job failed: {batch_prediction_job.error}")

        # Check the location of the output
        print(f"Job output location: {batch_prediction_job.output_location}")

        return batch_prediction_job.resource_name

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        job = BatchPredictionJob.get(batch_id)

        if job.state == BatchPredictionJob.State.JOB_STATE_SUCCEEDED:
            output_file_path = self._download_results(
                job.output_info.gcs_output_directory, benchmark_name, test_session_id
            )
            return self.process_batch_results(output_file_path)

        return None

    def cancel_batch(self, batch_id: str):
        job = BatchPredictionJob.get(batch_id)
        return job.cancel()

    def list_batches(self, limit: int = 10):
        return list(BatchPredictionJob.list(limit=limit))

    def process_batch_results(self, output_file_path: str) -> BatchResponse:
        results = []
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                response_data = data.get("predictions", [{}])[0]

                if "error" not in response_data:
                    response = response_data.get("content", "")
                    usage = Usage(
                        response_data.get("input_token_count", 0),
                        response_data.get("output_token_count", 0),
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
        input_file_path = (
            f"gs://{self.bucket_name}/{benchmark_name}/{test_session_id}/input.jsonl"
        )
        with open(input_file_path, "w") as f:
            for request in self.requests:
                f.write(json.dumps(request) + "\n")
        return input_file_path

    def _download_results(self, gcs_output_directory, benchmark_name, test_session_id):
        output_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_results.jsonl"
        if os.path.exists(output_file_path):
            return output_file_path

        # Implement GCS download logic here
        # For simplicity, assuming the file is already downloaded
        return output_file_path

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        return super().estimate_tokens_amount(messages)
