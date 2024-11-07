from ai_models.dto import BatchResponse
from ai_models.interfaces import BatchModelInterface


class BBHBatchRunner:
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
    ):
        model_name = model.get_model_name()

        # Check if benchmark already exists
        existing_batches = (
            self.batch_job_repo.get_by_test_session_and_benchmark_and_model(
                test_session_id, benchmark_name, model_name
            )
        )
        if existing_batches:
            print(
                f"Batch job already exists for Benchmark[{benchmark_name}], Session[{test_session_id}], model[{model_name}]"
            )
            return existing_batches[0].batch_id

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
                max_tokens=1000,  # BBH requires longer responses
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

        return batch_id

    def check_and_process_batch_results(
        self,
        batch_id: str,
        model: BatchModelInterface,
        model_results,
        prepared_questions,
        benchmark_name: str,
        test_session_id: int,
    ):
        batch_response = model.check_batch_results(
            benchmark_name, batch_id, test_session_id
        )
        if batch_response:
            self.process_batch_results(
                batch_response, model_results, prepared_questions, model
            )
            self.batch_job_repo.update_status(batch_id, "completed")
            return True
        else:
            print(f"Batch job {batch_id} not completed yet.\n")
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

                model_answer = self._extract_model_final_answer(item.response)
                score = self._calculate_score(
                    model_answer, prepared_question.correct_answer
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
                    execution_time=0,
                    score=score,
                )
            else:
                print(f"Warning: Could not process result for custom_id {custom_id}")

    def _calculate_score(self, model_answer: str, correct_answer: str) -> int:
        """Calculate score based on model response and correct answer."""
        correct_answer = correct_answer.strip().lower()

        # Check for exact match first
        if model_answer == correct_answer:
            return 1

        # print(
        #     f"Warning: No exact match - model_answer: {model_answer}, correct_answer: {correct_answer} - Doing further checks..."
        # )

        # For answers like (a), (b), etc - check if letter appears as standalone character
        if (
            len(correct_answer) == 3
            and correct_answer.startswith("(")
            and correct_answer.endswith(")")
        ):
            letter = correct_answer[1].lower()
            words = model_answer.split()
            if any(word == letter or word == f"({letter})" for word in words):
                return 1

        # For true/false questions
        if correct_answer in ["true", "false"]:
            if correct_answer in model_answer:
                return 1

        # print(
        #     f"Warning: All checks failed - model_answer: {model_answer}, correct_answer: {correct_answer}"
        # )
        return 0

    def _extract_model_final_answer(self, model_response: str) -> str:
        """Extract the final answer from model response."""
        model_answer = model_response.lower()
        final_answer_keyword = "####"

        # Extract final answer if delimiter exists
        if final_answer_keyword in model_answer:
            model_answer = model_answer.split(final_answer_keyword)[-1].strip()

        # Remove trailing period if exists
        if model_answer.endswith("."):
            model_answer = model_answer[:-1].strip()

        return model_answer
