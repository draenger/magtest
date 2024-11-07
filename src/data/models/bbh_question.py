from sqlalchemy import Column, Integer, String
from .base import Base


class BBHQuestion(Base):
    __tablename__ = "bbh_questions"

    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    category = Column(String, nullable=False)  # task type/category
    data_type = Column(String, nullable=False)  # 'train', 'test'
    explanation = Column(String, nullable=True)  # Optional explanation/reasoning
    helper_text = Column(String, nullable=True)  # Optional helper text

    def __repr__(self):
        return f"<BBHQuestion(id={self.id}, category='{self.category}', data_type='{self.data_type}')>"
