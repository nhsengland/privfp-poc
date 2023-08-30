from sqlalchemy import JSON, Column, DateTime, String, Text

from . import Base


class NER(Base):
    """
    Table for caching the API calls to the NER model
    The `id` column is a hash of the `input_text` column
    """

    __tablename__ = "ner"
    id = Column(String, primary_key=True)
    input_text = Column(Text)
    ner_response = Column(JSON)
    date_time = Column(DateTime)
