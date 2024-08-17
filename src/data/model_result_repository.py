from .repository import Repository
from .models import ModelResult


class ModelResultRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = ModelResult

    def add(
        self,
        model_name,
        benchmark_name,
        query,
        response,
        tokens_used,
        execution_time,
        score,
    ):
        entity = ModelResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            query=query,
            response=response,
            tokens_used=tokens_used,
            execution_time=execution_time,
            score=score,
        )
        try:
            super().add(entity)
            return entity
        except Exception as e:
            # Handle or log the exception as needed
            print(f"Error adding ModelResult: {e}")
            return None

    def get_by_model_name(self, model_name):
        session = self.db.get_session()
        results = (
            session.query(self.model).filter(self.model.model_name == model_name).all()
        )
        session.close()
        return results

    def get_by_benchmark(self, benchmark_name):
        session = self.db.get_session()
        results = (
            session.query(self.model)
            .filter(self.model.benchmark_name == benchmark_name)
            .all()
        )
        session.close()
        return results
