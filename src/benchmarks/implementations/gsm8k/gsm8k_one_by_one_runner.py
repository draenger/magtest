from time import perf_counter
from ai_models.interfaces import InstantModelInterface


class GSM8KOneByOneRunner:
    def __init__(self, model_result_repo):
        self.model_result_repo = model_result_repo

    def run_benchmark_one_by_one(
        self, prepared_questions, model_results, model: InstantModelInterface
    ):
        for model_result in model_results:
            prepared_question = prepared_questions.get(
                model_result.prepared_question_id
            )

            if not prepared_question:
                print(
                    f"Warning: PreparedQuestion not found for id {model_result.prepared_question_id}"
                )
                continue

            start_time = perf_counter()
            instant_response = model.predict(prepared_question.query)
            end_time = perf_counter()

            # Wyciągnij liczbę z odpowiedzi modelu
            model_answer = self._extract_number(instant_response.prediction)
            correct_answer = float(prepared_question.correct_answer)

            self.model_result_repo.update_execution_results(
                model_result.id,
                response=instant_response.prediction,
                actual_in_tokens=instant_response.usage.prompt_tokens,
                actual_out_tokens=instant_response.usage.completion_tokens,
                actual_in_cost=instant_response.usage.prompt_tokens
                * model.get_model_in_token_cost(),
                actual_out_cost=instant_response.usage.completion_tokens
                * model.get_model_out_token_cost(),
                execution_time=end_time - start_time,
                score=1 if model_answer == correct_answer else 0,
            )

    def _extract_number(self, response: str) -> float:
        """Wyciąga końcową odpowiedź liczbową z odpowiedzi modelu."""
        try:
            # Szukaj ostatniej linii zawierającej ####
            if "####" in response:
                final_answer = response.split("####")[-1].strip()
                # Konwertuj na float, usuwając wszystko poza liczbami i kropką
                return float(
                    "".join(c for c in final_answer if c.isdigit() or c == ".")
                )
            return float("".join(c for c in response if c.isdigit() or c == "."))
        except (ValueError, IndexError):
            print(f"Could not extract number from response: {response}")
            return float(
                "inf"
            )  # Zwróć wartość, która na pewno nie będzie równa poprawnej odpowiedzi
