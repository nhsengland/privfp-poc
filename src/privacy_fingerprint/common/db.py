import logging

# SQLite connection here
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from privacy_fingerprint.common.config import load_global_config
from privacy_fingerprint.common.models import LLM, NER, Base

session = None
logger = logging.getLogger(__name__)


def get_session():
    """Get a session for interaction with the database

    The location of the database file is set in the global config.
    If the value is empty then an in-memory database will be created"""
    global session
    if session is None:
        config = load_global_config()
        connection = "sqlite+pysqlite://"
        if config.cache.file_name != "":
            connection = f"{connection}/{config.cache.file_name}"
        else:
            logger.info(
                "Filename for the SQLite database is empty in the config file. "
                "An SQLite in-memory database is created instead of a file-based."
            )
        engine = create_engine(connection)
        session = sessionmaker(engine)
        table_names = inspect(engine).get_table_names()
        if (LLM.__tablename__ not in table_names) or (
            NER.__tablename__ not in table_names
        ):
            Base.metadata.create_all(engine)
    return session
