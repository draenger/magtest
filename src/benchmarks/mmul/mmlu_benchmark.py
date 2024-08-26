from benchmarks import Benchmark
from .mmlu_data_loader import MMULDataLoader
from time import perf_counter
import random


class MMULBenchmark(Benchmark):
    def __init__(self, model_result_repo, max_tests_per_benchmark=1, num_few_shot=5):
        self.model_result_repo = model_result_repo
        self.data = self.__load_data__(max_tests_per_benchmark)
        self.overall_correct = 0
        self.overall_total = 0
        self.results = {}
        self.num_few_shot = num_few_shot

    def __load_data__(self, max_tests_per_benchmark):
        data_loader = MMULDataLoader()
        mmlu_data = data_loader.process_data(max_tests_per_benchmark)

        if mmlu_data is None or "test" not in mmlu_data or "dev" not in mmlu_data:
            print(
                "Failed to load MMLU data. Please check the data source and try again."
            )
            return None
        return mmlu_data

    def __evaluate_model__(self, model, test_data, dev_data):
        correct = 0
        total = len(test_data)

        few_shot_template = """Question: {question}
Options: A. {A}, B. {B}, C. {C}, D. {D}
Answer: {answer}

"""

        main_question_template = """Question: {question}
Options: A. {A}, B. {B}, C. {C}, D. {D}
Answer with just the letter of the correct option (A, B, C, or D)."""

        for _, row in test_data.iterrows():
            few_shot_examples = dev_data.sample(n=min(self.num_few_shot, len(dev_data)))
            few_shot_prompt = "".join(
                few_shot_template.format(**example)
                for _, example in few_shot_examples.iterrows()
            )
            prompt = few_shot_prompt + main_question_template.format(**row)

            start_time = perf_counter()
            model_answer = model.predict(prompt)
            end_time = perf_counter()

            execution_time = end_time - start_time
            is_correct = model_answer == row["answer"]
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

    def run_benchmark(self, model):
        if self.data is None:
            print("No data loaded. Cannot run benchmark.")
            return None

        self.results = {}
        for category in self.data["test"]["category"].unique():
            category_test_data = self.data["test"][
                self.data["test"]["category"] == category
            ]
            category_dev_data = self.data["dev"][
                self.data["dev"]["category"] == category
            ]
            category_accuracy = self.__evaluate_model__(
                model, category_test_data, category_dev_data
            )
            self.results[category] = category_accuracy

        return self.get_overall_accuracy()

    def get_results(self):
        return self.results

    def get_overall_accuracy(self):
        return self.overall_correct / self.overall_total

    def get_calculated_results_accuracy(self):
        return sum(self.results.values()) / len(self.results)
