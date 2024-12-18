from ai_models.dto import BatchResponse
from ai_models.interfaces import BatchModelInterface


class MMLUBatchRunner:
    def __init__(self, model_result_repo, batch_job_repo):
        self.model_result_repo = model_result_repo
        self.batch_job_repo = batch_job_repo

    def run_benchmark_batch(
        self,
        prepared_questions,
        model_results,
        model: BatchModelInterface,
        test_session_id,
        benchmark_name,
        max_tokens: int,
    ):
        model_name = model.get_model_name()

        # Sprawdź, czy benchmark już istnieje
        existing_batches = (
            self.batch_job_repo.get_by_test_session_and_benchmark_and_model(
                test_session_id, benchmark_name, model_name
            )
        )
        if existing_batches:
            print(
                f"Batch job already exists for Benchmark[{benchmark_name}], Session[{test_session_id}], model[{model_name}]"
            )
            return existing_batches[0].batch_id  # Zwróć ID istniejącego batcha

        for model_result in model_results:
            prepared_question = prepared_questions.get(
                model_result.prepared_question_id
            )

            if not prepared_question:
                print(
                    f"Warning: PreparedQuestion not found for id {model_result.prepared_question_id}"
                )
                continue

            model.add_batch_request(
                custom_id=str(model_result.id),
                messages=[{"role": "user", "content": prepared_question.query}],
                max_tokens=max_tokens,
            )

        batch_ids = model.run_batch(
            benchmark_name,
            metadata={
                "description": f"Benchmark[{benchmark_name}], Session[{test_session_id}], model[{model_name}]"
            },
            test_session_id=test_session_id,
        )

        for batch_id in batch_ids:
            self.batch_job_repo.add(
                test_session_id=test_session_id,
                benchmark_name=benchmark_name,
                model_name=model_name,
                batch_id=batch_id,
            )
            print(
                f"Batch - Benchmark[{benchmark_name}], Session[{test_session_id}], model[{model_name}] created with ID: {batch_id}"
            )

        return batch_id

    def check_and_process_batch_results(
        self,
        batch_id: str,
        model: BatchModelInterface,
        model_results,
        prepared_questions,
        benchmark_name: str,
        test_session_id: int,
    ) -> bool:
        status = model.check_batch_status(batch_id)
        print(f"Batch status: {status}")

        if status == "completed":
            batch_response = model.process_batch_results(
                benchmark_name, batch_id, test_session_id
            )
            if batch_response:
                self.process_batch_results(
                    batch_response, model_results, prepared_questions, model
                )
                self.batch_job_repo.update_status(batch_id, "completed")
                return True
        elif status == "failed":
            print(f"Batch {batch_id} failed, attempting retry...")
            self.batch_job_repo.update_status(batch_id, "retry")
            new_batch_id = model.retry_batch(
                batch_id, metadata={"description": f"Retry of failed batch {batch_id}"}
            )
            if new_batch_id:
                self.batch_job_repo.add(
                    test_session_id=test_session_id,
                    benchmark_name=benchmark_name,
                    model_name=model.get_model_name(),
                    batch_id=new_batch_id,
                )
                print(f"Created new batch with ID: {new_batch_id}")

        return False

    def process_batch_results(
        self,
        batch_response: BatchResponse,
        model_results,
        prepared_questions,
        model: BatchModelInterface,
    ):
        for item in batch_response:
            custom_id = int(item.custom_id)
            model_result = next(
                (mr for mr in model_results if mr.id == custom_id),
                None,
            )

            if model_result:
                prepared_question = prepared_questions.get(
                    model_result.prepared_question_id
                )
                self.model_result_repo.update_execution_results(
                    model_result.id,
                    response=item.response,
                    actual_in_tokens=item.usage.prompt_tokens if item.usage else 0,
                    actual_out_tokens=item.usage.completion_tokens if item.usage else 0,
                    actual_in_cost=(
                        item.usage.prompt_tokens * model.get_model_in_token_cost()
                        if item.usage
                        else 0
                    ),
                    actual_out_cost=(
                        item.usage.completion_tokens * model.get_model_out_token_cost()
                        if item.usage
                        else 0
                    ),
                    execution_time=0,  # We don't have this information for batch processing
                    score=(
                        1
                        if item.response.lower()
                        == prepared_question.correct_answer.lower()
                        else 0
                    ),
                )
            else:
                print(f"Warning: Could not process result for custom_id {custom_id}")
