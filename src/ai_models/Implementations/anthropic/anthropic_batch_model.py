from ...base_batch_model import BaseBatchModel
from ...dto.batch_response import BatchResponse, BatchResponseItem
from ...dto.usage import Usage
from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
import os
from dotenv import load_dotenv
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
        load_dotenv()
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model_name, max_tokens=max_tokens, messages=messages
            ),
        )
        self.requests.append(request)

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        message_batch = self.client.beta.messages.batches.create(requests=self.requests)
        return [message_batch.id]

    def check_batch_status(self, batch_id: str) -> Optional[str]:
        try:
            message_batch = self.client.beta.messages.batches.retrieve(batch_id)
            if message_batch.processing_status == "ended":
                return "completed"
            elif message_batch.processing_status == "failed":
                return "failed"
            return "in_progress"
        except Exception as e:
            print(f"Error checking batch status: {str(e)}")
            return None

    def cancel_batch(self, batch_id: str):
        return self.client.beta.messages.batches.cancel(batch_id)

    def list_batches(self, limit: int = 10):
        return self.client.beta.messages.batches.list(limit=limit)

    def process_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        try:
            message_batch = self.client.beta.messages.batches.retrieve(batch_id)
            if message_batch.processing_status != "ended":
                return None

            results = []
            for result in self.client.beta.messages.batches.results(batch_id):
                custom_id = result.custom_id
                result_data = result.result

                if result_data.type == "succeeded":
                    message = result_data.message
                    response = message.content[0].text if message.content else None
                    usage = Usage(
                        message.usage.input_tokens, message.usage.output_tokens
                    )
                    status = "success"
                else:
                    response = None
                    usage = None
                    status = result_data.type

                results.append(
                    BatchResponseItem(
                        custom_id=custom_id,
                        response=response,
                        usage=usage,
                        status=status,
                    )
                )

            return BatchResponse(results)
        except Exception as e:
            print(f"Error processing batch results: {str(e)}")
            return None

    def estimate_tokens_amount(self, messages: List[dict]) -> int:
        total_tokens = 0
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                total_tokens += super().estimate_tokens_amount(message["content"])
            else:
                print(f"Warning: Unexpected message format: {message}")
        return total_tokens

    def retry_batch(
        self,
        batch_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        print("Retry not implemented for Anthropic Batch Model")
        return None

    def get_input_file_url(self, batch_id: str) -> Optional[str]:
        print("Get input file URL not implemented for Anthropic Batch Model")
        return None
