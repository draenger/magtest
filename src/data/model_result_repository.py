from .repository import Repository
from .models import ModelResult

class ModelResultRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = ModelResult

    def get_by_model_name(self, model_name):
        session = self.db.get_session()
        results = session.query(self.model).filter(self.model.model_name == model_name).all()
        session.close()
        return results

    def get_by_benchmark(self, benchmark_name):
        session = self.db.get_session()
        results = session.query(self.model).filter(self.model.benchmark_name == benchmark_name).all()
        session.close()
        return results
