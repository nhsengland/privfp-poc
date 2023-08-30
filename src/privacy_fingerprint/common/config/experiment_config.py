from typing import List

from pydantic import BaseModel


class ExperimentSyntheaConfig(BaseModel):
    county: str
    encounter_type: str
    num_records: int
    extra_config: dict
    records_per_patient: int
    ethnicity_types: List[str]


class ExperimentOpenAPIConfig(BaseModel):
    model: str
    max_tokens: int
    temperature: float
    prompt: str


class ScoringConfig(BaseModel):
    encoding_scheme: str
    max_columns: int


class ExperimentConfig(BaseModel):
    synthea: ExperimentSyntheaConfig
    openai: ExperimentOpenAPIConfig
    scoring: ScoringConfig
