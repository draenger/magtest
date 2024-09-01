from .repository import Repository
from data.models import ModelResult, PreparedQuestion
from sqlalchemy import and_
from datetime import datetime


class ModelResultRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = ModelResult

    def add(self, **kwargs):
        entity = ModelResult(**kwargs)
        try:
            session = self.db.get_session()
            session.add(entity)
            session.commit()
            session.close()
            return entity
        except Exception as e:
            session.rollback()
            print(f"Error adding ModelResult: {e}")
            return None
        finally:
            session.close()

    def update_execution_results(self, model_result_id, **kwargs):
        session = self.db.get_session()
        try:
            model_result = (
                session.query(ModelResult)
                .filter(ModelResult.id == model_result_id)
                .first()
            )
            if model_result:
                for key, value in kwargs.items():
                    setattr(model_result, key, value)
                model_result.execution_date = datetime.utcnow()
                model_result.status = "completed"
                session.commit()
            session.close()
            return model_result
        except Exception as e:
            session.rollback()
            print(f"Error updating ModelResult: {e}")
            return None
        finally:
            session.close()

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
            .join(PreparedQuestion)
            .filter(PreparedQuestion.benchmark_name == benchmark_name)
            .all()
        )
        session.close()
        return results

    def get_results_for_session_and_model(self, test_session_id, model_name):
        session = self.db.get_session()
        try:
            results = (
                session.query(self.model)
                .join(
                    PreparedQuestion,
                    self.model.prepared_question_id == PreparedQuestion.id,
                )
                .filter(
                    and_(
                        PreparedQuestion.test_session_id == test_session_id,
                        self.model.model_name == model_name,
                    )
                )
                .all()
            )
            return results
        except Exception as e:
            print(f"Error getting results for session and model: {e}")
            return []
        finally:
            session.close()
