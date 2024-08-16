from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ModelResult(Base):
    __tablename__ = 'model_results'

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    benchmark_name = Column(String, nullable=False)
    query = Column(String, nullable=False)
    response = Column(String, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    execution_time = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelResult(model_name='{self.model_name}', benchmark_name='{self.benchmark_name}', score={self.score})>"
