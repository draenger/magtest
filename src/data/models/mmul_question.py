from sqlalchemy import Column, Integer, String
from .base import Base


class MMULQuestion(Base):
    __tablename__ = "mmul_questions"

    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    option_a = Column(String, nullable=False)
    option_b = Column(String, nullable=False)
    option_c = Column(String, nullable=False)
    option_d = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    subcategory = Column(String, nullable=False)
    category = Column(String, nullable=False)
    group = Column(String, nullable=False)
    data_type = Column(String, nullable=False)

    def __repr__(self):
        return f"<MMULQuestion(id={self.id}, question='{self.question[:20]}...', data_type='{self.data_type}')>"
