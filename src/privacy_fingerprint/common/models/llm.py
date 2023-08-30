from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from . import Base


class LLM(Base):
    """
    Table for caching the API calls to the LLM model
    The `id` column is a hash of the combination of the following columns:
    `prompt`, `encounter`, `max_tokens`, `temperature`, `model`
    """

    __tablename__ = "llm"
    id = Column(String, primary_key=True)
    prompt = Column(String)
    encounter = Column(JSON)
    max_tokens = Column(Integer)
    temperature = Column(Float)
    model = Column(String)
    llm_response = Column(Text)
    date_time = Column(DateTime)
