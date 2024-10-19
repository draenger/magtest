from .repository import Repository
from data.models import BatchJob
from sqlalchemy import and_
from datetime import datetime


class BatchJobRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = BatchJob

    def add(
        self, test_session_id, benchmark_name, model_name, batch_id, status="pending"
    ):
        entity = BatchJob(
            test_session_id=test_session_id,
            benchmark_name=benchmark_name,
            model_name=model_name,
            batch_id=batch_id,
            status=status,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        try:
            session = self.db.get_session()
            session.add(entity)
            session.commit()
            session.close()
            return entity
        except Exception as e:
            session.rollback()
            print(f"Error adding BatchJob: {e}")
            return None
        finally:
            session.close()

    def update_status(self, batch_id, status):
        session = self.db.get_session()
        try:
            batch_job = (
                session.query(BatchJob).filter(BatchJob.batch_id == batch_id).first()
            )
            if batch_job:
                batch_job.status = status
                batch_job.updated_at = datetime.utcnow()
                session.commit()
            session.close()
            return batch_job
        except Exception as e:
            session.rollback()
            print(f"Error updating BatchJob status: {e}")
            return None
        finally:
            session.close()

    def get_pending_jobs(self):
        session = self.db.get_session()
        try:
            pending_jobs = (
                session.query(BatchJob).filter(BatchJob.status == "pending").all()
            )
            return pending_jobs
        except Exception as e:
            print(f"Error getting pending BatchJobs: {e}")
            return []
        finally:
            session.close()

    def get_job_status(self, batch_id):
        session = self.db.get_session()
        try:
            batch_job = (
                session.query(BatchJob).filter(BatchJob.batch_id == batch_id).first()
            )
            return batch_job.status if batch_job else None
        except Exception as e:
            print(f"Error getting BatchJob status: {e}")
            return None
        finally:
            session.close()

    def get_by_test_session_and_benchmark_and_model(
        self, test_session_id, benchmark_name, model_name
    ):
        session = self.db.get_session()
        try:
            batch_jobs = (
                session.query(BatchJob)
                .filter(
                    and_(
                        BatchJob.test_session_id == test_session_id,
                        BatchJob.benchmark_name == benchmark_name,
                        BatchJob.model_name == model_name,
                    )
                )
                .all()
            )
            return batch_jobs
        except Exception as e:
            print(f"Error getting BatchJobs by test session, benchmark, and model: {e}")
            return []
        finally:
            session.close()
