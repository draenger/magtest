# google_batch_model.py
from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
import vertexai
from vertexai.preview.batch_prediction import BatchPredictionJob
from google.oauth2 import service_account
from google.cloud import storage
import os
import json
import time
from typing import List, Optional
import ftfy


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
        # Get environment variables
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION")
        self.bucket = os.getenv("GOOGLE_CLOUD_BUCKET")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not all(
            [
                self.project_id,
                self.location,
                self.bucket,
                credentials_path,
            ]
        ):
            raise ValueError("Missing required environment variables")

        # Initialize credentials
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as e:
            raise Exception(f"Failed to initialize credentials: {str(e)}")

        # Initialize Vertex AI
        try:
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials,
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Vertex AI: {str(e)}")

        # Initialize storage client
        try:
            self.storage_client = storage.Client(credentials=self.credentials)
        except Exception as e:
            raise Exception(f"Failed to initialize storage client: {str(e)}")

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = {
            "request": {
                "contents": self._convert_messages_to_prompt(messages),
                "generationConfig": {"maxOutputTokens": max_tokens},
                "labels": {"custom_id": f"google_{custom_id}"},
            }
        }
        self.requests.append((custom_id, request))

    def _convert_messages_to_prompt(self, messages: List[dict]) -> List[dict]:
        return [
            {
                "parts": [{"text": msg["content"]}],
                "role": msg["role"],
            }
            for msg in messages
        ]

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        print("Running batch job")
        try:
            salt = os.urandom(8).hex()
            input_file_path = self._create_local_input_file(
                benchmark_name, test_session_id, salt
            )
            input_uri = self._upload_to_cloud_storage(input_file_path, test_session_id)
            output_uri = (
                f"gs://{self.bucket}/outputs/{self.model_name}_batch_responses_{salt}"
            )

            batch_prediction_job = BatchPredictionJob.submit(
                source_model=self.model_name,
                input_dataset=input_uri,
                output_uri_prefix=output_uri,
            )
            print(f"Batch job submitted with ID: {batch_prediction_job.resource_name}")
            return [batch_prediction_job.resource_name]
        except Exception as e:
            raise Exception(f"Failed to run batch job: {str(e)}")

    def _create_local_input_file(
        self, benchmark_name: str, test_session_id: int, salt: str
    ) -> str:
        input_file_path = f"batch/{test_session_id}/{benchmark_name}/{self.model_name}_batch_requests_{salt}.jsonl"
        try:
            os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
            with open(input_file_path, "w") as f:
                for _, request in self.requests:
                    f.write(json.dumps(request) + "\n")
            return input_file_path
        except Exception as e:
            raise Exception(f"Failed to create input file: {str(e)}")

    def _upload_to_cloud_storage(
        self, input_file_path: str, test_session_id: int
    ) -> str:
        try:
            file_name = input_file_path.split("/")[-1]
            cloud_path = f"inputs/{file_name}"
            input_uri = f"gs://{self.bucket}/{cloud_path}"

            bucket = self.storage_client.bucket(self.bucket)
            blob = bucket.blob(cloud_path)
            blob.upload_from_filename(input_file_path)

            return input_uri
        except Exception as e:
            raise Exception(f"Failed to upload to cloud storage: {str(e)}")

    def check_batch_status(self, batch_id: str) -> Optional[str]:
        try:
            job = BatchPredictionJob(batch_id)
            if job.has_ended:
                return "completed" if job.has_succeeded else "failed"
            return "in_progress"
        except Exception as e:
            print(f"Error checking batch status: {str(e)}")
            return None

    def process_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        try:
            job = BatchPredictionJob(batch_id)
            if not job.has_succeeded:
                return None

            results = self._read_results_from_storage(job.output_location)
            return self._process_results(results)
        except Exception as e:
            print(f"Error processing batch results: {str(e)}")
            return None

    def _read_results_from_storage(self, output_location: str) -> List[dict]:
        try:
            results = []
            bucket_name = self.bucket
            prefix = "/".join(output_location.replace("gs://", "").split("/")[1:])

            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                if blob.name.endswith(".jsonl"):
                    content = blob.download_as_text(encoding="utf-8")
                    current_object = ""
                    chunk_prefix = '{"status":'
                    # Split content into chunks starting with prefix
                    chunks = content.split(chunk_prefix)
                    chunks = [chunk for chunk in chunks if chunk.strip()]
                    for chunk in chunks:
                        try:
                            # Reconstruct the object by adding the prefix back
                            current_object = chunk_prefix + chunk
                            json_obj = json.loads(current_object)
                            results.append(json_obj)

                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON object: {e}")
                            print(f"Problematic chunk start: {current_object[:100]}...")
                            continue
            return results
        except Exception as e:
            print(f"Error reading from storage: {str(e)}")
            raise Exception(f"Failed to read results from storage: {str(e)}")

    def _process_results(self, results) -> BatchResponse:
        try:
            processed_results = []
            for result in results:
                # Pobierz custom_id z labels w request
                custom_id = (
                    result.get("request", {})
                    .get("labels", {})
                    .get("custom_id")
                    .split("_")[1]
                )

                # Pobierz odpowiedź z pierwszego kandydata
                response_content = (
                    result.get("response", {})
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )

                # Pobierz informacje o użyciu tokenów
                usage_metadata = result.get("response", {}).get("usageMetadata", {})

                response_item = BatchResponseItem(
                    custom_id=custom_id,
                    response=response_content.strip(),
                    usage=Usage(
                        prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                        completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    ),
                    status="success" if response_content else "failed",
                )
                processed_results.append(response_item)
            return BatchResponse(processed_results)
        except Exception as e:
            raise Exception(f"Failed to process results: {str(e)}")

    def retry_batch(
        self,
        batch_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        print("Retry not implemented for Google Batch Model")
        return None

    def get_input_file_url(self, batch_id: str) -> Optional[str]:
        print("Get input file URL not implemented for Google Batch Model")
        return None
