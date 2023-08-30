from unittest.mock import Mock

from pytest import fixture

from privacy_fingerprint.common.config import (
    load_experiment_config,
    load_global_config,
)
from privacy_fingerprint.generate.language_model import LMGenerator


@fixture
def temp_global_cache():
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


@fixture
def temp_experiment_cache():
    return {
        "synthea": {
            "county": "Hampshire",
            "encounter_type": "Emergency room admission (procedure)",
            "num_records": 100,
            "extra_config": {},
            "records_per_patient": 1,
            "ethnicity_types": [
                "White - British",
                "White - Irish",
                "White - Any other White background",
                "Mixed - White and Black Caribbean",
                "Mixed - White and Black African",
                "Mixed - White and Asian",
                "Mixed - Any other mixed background",
                "Asian or Asian British - Indian",
                "Asian or Asian British - Pakistani",
                "Asian or Asian British - Bangladeshi",
                "Asian or Asian British - Any other Asian background",
                "Black or Black British - Caribbean",
                "Black or Black British - African",
                "Black or Black British - Any other Black background",
                "Other Ethnic Groups - Chinese",
                "Other Ethnic Groups - Any other ethnic group",
            ],
        },
        "openai": {
            "model": "text-davinci-003",
            "max_tokens": 256,
            "temperature": 0.7,
            "prompt": "Describe this patient as if you were a medical doctor.",
        },
        "scoring": {
            "encoding_scheme": "one-hot",
            "max_columns": 30,
        },
    }


def test_cache_is_used(temp_global_cache, temp_experiment_cache):
    load_global_config(temp_global_cache)
    load_experiment_config(temp_experiment_cache)
    generator = LMGenerator()
    mock_call_api = Mock(side_effect=lambda x: [f"Record {i}" for i in x])
    generator._call_api = mock_call_api
    queries = ["a", "b"]
    response1 = list(generator.generate_text(queries))
    assert len(response1) == len(queries)
    assert mock_call_api.call_count == 1
    response2 = list(generator.generate_text(queries))
    assert len(response2) == len(queries)
    assert mock_call_api.call_count == 1
