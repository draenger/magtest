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

        selected_columns = [
            "benchmark_name",
            "model_name",
            "queries",
            "avg_score",
            "avg_execution_time",
            "est_tokens",
            "act_tokens",
            "est_cost",
            "act_cost",
        ]

        df["total_execution_time"] = df["avg_execution_time"] * df["queries"]

        result_df = df[selected_columns].copy()

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
