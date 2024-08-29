from benchmarks import Benchmark
from time import perf_counter


class MMULBenchmark(Benchmark):
    def __init__(self, model_result_repo, test_preparation):
        self.model_result_repo = model_result_repo
        self.test_preparation = test_preparation
        self.results = {}
        self.overall_correct = 0
        self.overall_total = 0

    def run_benchmark(self, model, max_tests_per_benchmark, num_few_shot):
        prepared_data, test_session_id = self.test_preparation.prepare_test_data(
            max_tests_per_benchmark, num_few_shot
        )

        for test_case in prepared_data:
            # Calculate estimated tokens and costs
            estimated_in_tokens = len(
                test_case["prompt"].split()
            )  # This is a very naive estimation
            estimated_out_tokens = 1  # Assuming single letter response
            estimated_in_cost = estimated_in_tokens * 0.001  # Dummy cost calculation
            estimated_out_cost = estimated_out_tokens * 0.002  # Dummy cost calculation

            model_result = self.model_result_repo.add(
                prepared_question_id=test_case["id"],
                model_name=model.model_name,
                estimated_in_tokens=estimated_in_tokens,
                estimated_out_tokens=estimated_out_tokens,
                estimated_in_cost=estimated_in_cost,
                estimated_out_cost=estimated_out_cost,
            )

            start_time = perf_counter()
            model_answer = model.predict(test_case["prompt"])
            end_time = perf_counter()

            execution_time = end_time - start_time
            is_correct = model_answer == test_case["answer"]
            score = 1 if is_correct else 0

            self.overall_correct += score
            self.overall_total += 1

            if test_case["category"] not in self.results:
                self.results[test_case["category"]] = {"correct": 0, "total": 0}
            self.results[test_case["category"]]["correct"] += score
            self.results[test_case["category"]]["total"] += 1

            # Calculate actual tokens and costs
            actual_in_tokens = len(
                test_case["prompt"].split()
            )  # This is a very naive calculation
            actual_out_tokens = len(model_answer)
            actual_in_cost = actual_in_tokens * 0.001  # Dummy cost calculation
            actual_out_cost = actual_out_tokens * 0.002  # Dummy cost calculation

            self.model_result_repo.update_execution_results(
                model_result.id,
                response=model_answer,
                actual_in_tokens=actual_in_tokens,
                actual_out_tokens=actual_out_tokens,
                actual_in_cost=actual_in_cost,
                actual_out_cost=actual_out_cost,
                execution_time=execution_time,
                score=score,
            )

        return self.get_overall_accuracy()

    # ... (other methods remain the same)
