from unittest.mock import Mock

import pytest
from pytest import fixture

import pandas as pd

from privacy_fingerprint.common.config import load_global_config
from privacy_fingerprint.extract.aws_comprehend import (
    ComprehendExtractor,
    calculate_ner_cost,
)


@fixture
def temp_cache():
    return {
        "synthea": {"install_directory": ""},
        "openai": {
            "api_key": "",
            "batch_size": 20,
            "delay_on_error": 1,
            "backoff_on_error": 2,
            "max_delay_on_error": 360,
            "retry_attempts": 10,
        },
        "comprehendmedical": {
            "profile": "default",
            "service": "comprehendmedical",
        },
        "cache": {"file_name": ""},
        "pcm": {"path_to_julia": ""},
    }


def test_cache_is_used(temp_cache):
    load_global_config(temp_cache)

    class TestingComprehendExtractor(ComprehendExtractor):
        def __init__(self):
            pass

    extractor = TestingComprehendExtractor()
    mock_call_api = Mock(side_effect=lambda x: f"NER {x}")
    extractor._call_api = mock_call_api
    query = "This is a test"
    list(extractor.extract_record(query))
    assert mock_call_api.call_count == 1
    list(extractor.extract_record(query))
    assert mock_call_api.call_count == 1


def test_calculate_ner_cost_df():
    records = pd.DataFrame(
        data=[["how expensive is AWS"], ["let's find out"]], columns=["text"]
    )
    cost = calculate_ner_cost(records, "text")
    assert cost == pytest.approx(0.0034)


def test_calculate_ner_cost_list():
    records = ["how expensive is AWS", "let's find out"]
    cost = calculate_ner_cost(records, "text")
    assert cost == pytest.approx(0.0034)
