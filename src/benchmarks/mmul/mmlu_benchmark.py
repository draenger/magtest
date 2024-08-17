from benchmarks import Benchmark
from .mmlu_data_loader import MMULDataLoader
from time import perf_counter


class MMULBenchmark(Benchmark):
    def __init__(
        self, model_result_repo, data_set_dir="test", max_tests_per_benchmark=1
    ):
        self.model_result_repo = model_result_repo
        self.data = self.__load_data__(data_set_dir, max_tests_per_benchmark)
        self.overall_correct = 0
        self.overall_total = 0
        self.results = {}

    def __evaluate_model__(self, model, data):
        correct = 0
        total = len(data)

        for _, row in data.iterrows():
            question = row["question"]
            options = [row["A"], row["B"], row["C"], row["D"]]
            correct_answer = row["answer"]
            prompt = f"Question: {question}\nOptions: A. {options[0]}, B. {options[1]}, C. {options[2]}, D. {options[3]}\nAnswer with just the letter of the correct option (A, B, C, or D)."

            start_time = perf_counter()
            model_answer = model.predict(prompt)
            end_time = perf_counter()

            execution_time = end_time - start_time
            is_correct = model_answer == correct_answer
            score = 1 if is_correct else 0
            correct += score

            self.model_result_repo.add(
                model_name=model.model_name,
                benchmark_name="MMLU",
                query=prompt,
                response=model_answer,
                tokens_used=model.get_tokens_used(),
                execution_time=execution_time,
                score=score,
            )

        self.overall_correct += correct
        self.overall_total += total
        accuracy = correct / total
        return accuracy

    def __load_data__(self, data_set_dir, max_tests_per_benchmark):
        data_loader = MMULDataLoader(data_set_dir=data_set_dir)
        mmlu_data = data_loader.process_data(max_tests_per_benchmark)

        if mmlu_data is None:
            print(
                "Failed to load MMLU data. Please check the data source and try again."
            )
            return None
        return mmlu_data

    def run_benchmark(self, model):
        if self.data is None:
            print("No data loaded. Cannot run benchmark.")
            return None

        self.results = {}
        for category in self.data["category"].unique():
            category_data = self.data[self.data["category"] == category]
            category_accuracy = self.__evaluate_model__(model, category_data)
            self.results[category] = category_accuracy

        return self.get_overall_accuracy()

    def get_results(self):
        return self.results

    def get_overall_accuracy(self):
        return self.overall_correct / self.overall_total

    def get_calculated_results_accuracy(self):
        return sum(self.results.values()) / len(self.results)
