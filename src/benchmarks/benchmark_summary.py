from sqlalchemy import func
import pandas as pd
from data.models import PreparedQuestion, ModelResult


class BenchmarkSummary:
    def __init__(self, db):
        self.db = db

    def _get_base_query(self, test_session_id):
        return (
            self.db.get_session()
            .query(
                PreparedQuestion.benchmark_name,
                PreparedQuestion.category,
                ModelResult.model_name,
            )
            .join(
                PreparedQuestion,
                ModelResult.prepared_question_id == PreparedQuestion.id,
            )
            .filter(PreparedQuestion.test_session_id == test_session_id)
        )

    def _add_common_metrics(self, query):
        return query.add_columns(
            func.count(ModelResult.id).label("queries"),
            func.avg(ModelResult.score).label("avg_score"),
            func.sum(ModelResult.estimated_in_tokens).label("est_in_tokens"),
            func.sum(ModelResult.estimated_out_tokens).label("est_out_tokens"),
            func.sum(ModelResult.actual_in_tokens).label("act_in_tokens"),
            func.sum(ModelResult.actual_out_tokens).label("act_out_tokens"),
            func.sum(ModelResult.estimated_in_cost).label("est_in_cost"),
            func.sum(ModelResult.estimated_out_cost).label("est_out_cost"),
            func.sum(ModelResult.actual_in_cost).label("act_in_cost"),
            func.sum(ModelResult.actual_out_cost).label("act_out_cost"),
            func.avg(ModelResult.execution_time).label("avg_execution_time"),
        )

    def _execute_query(self, query):
        try:
            results = query.all()
            df = pd.DataFrame(results)
            df["est_tokens"] = df["est_in_tokens"] + df["est_out_tokens"]
            df["act_tokens"] = df["act_in_tokens"] + df["act_out_tokens"]
            df["est_cost"] = df["est_in_cost"] + df["est_out_cost"]
            df["act_cost"] = df["act_in_cost"] + df["act_out_cost"]
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
            self.db.get_session().close()

    def get_detailed_summary(self, test_session_id):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.group_by(
            PreparedQuestion.benchmark_name,
            PreparedQuestion.category,
            ModelResult.model_name,
        )
        return self._execute_query(query)

    def get_benchmark_summary(self, test_session_id):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.group_by(PreparedQuestion.benchmark_name, ModelResult.model_name)
        df = self._execute_query(query)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Benchmark",
                    "Model",
                    "Queries",
                    "Avg Score",
                    "Total Execution Time",
                    "Est Tokens",
                    "Act Tokens",
                    "Est Cost",
                    "Act Cost",
                ]
            )

        df["total_execution_time"] = df.get("avg_execution_time", 0) * df["queries"]
        df["est_tokens"] = df["est_in_tokens"] + df["est_out_tokens"]
        df["act_tokens"] = df["act_in_tokens"] + df["act_out_tokens"]
        df["est_cost"] = df["est_in_cost"] + df["est_out_cost"]
        df["act_cost"] = df["act_in_cost"] + df["act_out_cost"]

        result_df = df[
            [
                "benchmark_name",
                "model_name",
                "queries",
                "avg_score",
                "total_execution_time",
                "est_tokens",
                "act_tokens",
                "est_cost",
                "act_cost",
            ]
        ].copy()

        result_df.columns = [
            "Benchmark",
            "Model",
            "Queries",
            "Avg Score",
            "Total Execution Time",
            "Est Tokens",
            "Act Tokens",
            "Est Cost",
            "Act Cost",
        ]

        return result_df

    def get_model_summary(self, test_session_id, model_name):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.filter(ModelResult.model_name == model_name)
        query = query.group_by(PreparedQuestion.benchmark_name)
        return self._execute_query(query)

    def get_category_summary(self, test_session_id, benchmark_name):
        query = self._get_base_query(test_session_id)
        query = self._add_common_metrics(query)
        query = query.filter(PreparedQuestion.benchmark_name == benchmark_name)
        query = query.group_by(PreparedQuestion.category, ModelResult.model_name)
        return self._execute_query(query)

    def print_detailed_summary(self, test_session_id):
        df = self.get_detailed_summary(test_session_id)
        print("Detailed Benchmark Summary")
        print("==========================")
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_benchmark_summary(self, test_session_id):
        df = self.get_benchmark_summary(test_session_id)
        print("Benchmark Summary")
        print("=================")
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=20,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_model_summary(self, test_session_id, model_name):
        df = self.get_model_summary(test_session_id, model_name)
        print(f"Model Summary: {model_name}")
        print("=" * (16 + len(model_name)))
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_category_summary(self, test_session_id, benchmark_name):
        df = self.get_category_summary(test_session_id, benchmark_name)
        print(f"Category Summary: {benchmark_name}")
        print("=" * (18 + len(benchmark_name)))
        print(
            df.to_string(
                index=False,
                justify="center",
                col_space=15,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print("\n")

    def print_full_summary(self, test_session_id):
        print("Full Benchmark Summary")
        print("======================")
        self.print_benchmark_summary(test_session_id)

        print("Detailed Summaries")
        print("==================")
        df = self.get_benchmark_summary(test_session_id)
        for _, row in df.iterrows():
            benchmark = row["Benchmark"]
            model = row["Model"]
            print(f"\nBenchmark: {benchmark}, Model: {model}")
            print("-" * (len(benchmark) + len(model) + 22))
            self.print_category_summary(test_session_id, benchmark)

    def format_float(self, value):
        return f"{value:.4f}" if pd.notnull(value) else "N/A"

    def get_summary_string(self, test_session_id):
        df = self.get_benchmark_summary(test_session_id)
        summary_string = "Benchmark Summary\n=================\n"
        summary_string += df.to_string(
            index=False,
            justify="center",
            col_space=20,
            formatters={
                col: self.format_float
                for col in df.select_dtypes(include=["float64"]).columns
            },
        )
        return summary_string

    def get_detailed_summary_string(self, test_session_id):
        df = self.get_detailed_summary(test_session_id)
        summary_string = "Detailed Benchmark Summary\n==========================\n"
        summary_string += df.to_string(
            index=False,
            justify="center",
            col_space=15,
            formatters={
                col: self.format_float
                for col in df.select_dtypes(include=["float64"]).columns
            },
        )
        return summary_string
