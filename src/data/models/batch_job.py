from sqlalchemy import Column, Integer, String, DateTime
from .base import Base


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True)
    test_session_id = Column(String, nullable=False)
    benchmark_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    batch_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<BatchJob(id={self.id}, test_session_id='{self.test_session_id}', benchmark_name='{self.benchmark_name}', model_name='{self.model_name}', batch_id='{self.batch_id}', status='{self.status}')>"
