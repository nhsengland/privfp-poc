import hashlib
import json

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# These must follow the Base declaration
from .llm import LLM  # noqa: E402
from .ner import NER  # noqa: E402

__all__ = [
    "LLM",
    "NER",
    "Base",
    "generate_id",
]


def generate_id(**kwargs):
    """Generate a unique hash from API inputs

    Accepts any arguments and produces a consistent hash regardless of order

    :returns: Hash in hexadecimal format"""
    hash = hashlib.sha1()
    hash.update(json.dumps(kwargs, sort_keys=True).encode("utf-8"))
    return hash.hexdigest()
