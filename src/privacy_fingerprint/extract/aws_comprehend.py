from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import boto3
import pandas as pd

from privacy_fingerprint.common import Record
from privacy_fingerprint.common.config import load_global_config
from privacy_fingerprint.common.db import get_session
from privacy_fingerprint.common.models import NER, generate_id

from . import Extractor


class ComprehendExtractor(Extractor):
    """An extractor using AWS comprehend"""

    def __init__(self):
        config = load_global_config()
        self.session = boto3.Session(
            profile_name=config.comprehendmedical.profile
        )
        self.comprehend_client = self.session.client(
            config.comprehendmedical.service
        )

    def extract_record(self, clinical_text: str) -> Dict:
        text_id = generate_id(input_text=clinical_text)
        with get_session()() as session:
            result = session.query(NER).where(NER.id == text_id).one_or_none()
            if result is None:
                api_result = self._call_api(clinical_text)
                result = NER(
                    id=text_id,
                    input_text=clinical_text,
                    ner_response=api_result,
                    date_time=datetime.now(),
                )
                session.add(result)
                session.commit()
            return result.ner_response

    def _call_api(self, clinical_text: str) -> Dict:
        return self.comprehend_client.detect_entities_v2(Text=clinical_text)


def _extract_entity_type(record, entity_type):
    return [i for i in record if i["Type"] == entity_type]


def _extract_entity_category(record, category_type):
    return [i for i in record if i["Category"] == category_type]


def _sorted_mention(entities, key):
    freq = defaultdict(int)
    for entity in entities:
        freq[entity[key]] += 1
    if len(freq) == 0:
        return None
    return [
        i[0] for i in sorted(freq.items(), key=lambda x: x[1], reverse=True)
    ]


def extract_nhs_number(record):
    id_candidate = " ".join(
        [i["Text"] for i in _extract_entity_type(record, "ID")]
    )
    # NHS numbers look like US phone numbers and can be mistaken as such
    # We assume records will not have phone numbers and the UK format is different
    phone_candidate = " ".join(
        [i["Text"] for i in _extract_entity_type(record, "PHONE_OR_FAX")]
    )
    if len(id_candidate) > len(phone_candidate):
        return id_candidate
    if len(phone_candidate) == 0:
        return None
    return phone_candidate


def extract_name(record):
    candidates = _extract_entity_type(record, "NAME")
    if len(candidates) == 0:
        return None
    return _sorted_mention(candidates, "Text")[0]


def extract_date_of_birth(record):
    candidates = _extract_entity_type(record, "DATE")
    if len(candidates) == 0:
        return None
    filtered_candidates = [
        candidate
        for candidate in candidates
        if pd.notnull(pd.to_datetime(candidate["Text"], errors="coerce"))
    ]
    if len(filtered_candidates) == 0:
        return None
    else:
        return sorted(
            filtered_candidates,
            key=lambda candidate: pd.to_datetime(
                candidate["Text"], errors="coerce"
            ),
            reverse=False,
        )[0]["Text"]


def extract_disease(record):
    candidates = _extract_entity_type(record, "DX_NAME")
    return set([i["Text"].lower() for i in candidates])


def extract_symptoms(record):
    candidates = _extract_entity_category(record, "MEDICAL_CONDITION")
    return set([i["Text"].lower() for i in candidates])


def extract_date_of_visit(record):
    candidates = _extract_entity_type(record, "DATE")
    if len(candidates) == 0:
        return None
    filtered_candidates = [
        candidate
        for candidate in candidates
        if pd.notnull(pd.to_datetime(candidate["Text"], errors="coerce"))
    ]
    if len(filtered_candidates) == 0:
        return None
    else:
        return sorted(
            filtered_candidates,
            key=lambda candidate: pd.to_datetime(
                candidate["Text"], errors="coerce"
            ),
            reverse=True,
        )[0]["Text"]


def extract_department(record):
    candidates = _extract_entity_type(record, "ADDRESS")
    for candidate in candidates:
        for likely_text in ["hospital", "surgery", "clinic", "infirmary"]:
            if likely_text in candidate["Text"].lower():
                return candidate["Text"]
    if len(candidates) == 0:
        return None
    return _sorted_mention(candidates, "Text")[0]


def extract_gender(record):
    candidates = _extract_entity_type(record, "GENDER")
    if len(candidates) == 0:
        return None
    return _sorted_mention(candidates, "Text")[0]


def extract_ethnicity(record):
    candidates = _extract_entity_type(record, "RACE_ETHNICITY")
    return " ".join([i["Text"] for i in candidates])


def extract_treatment(record):
    candidates = _extract_entity_type(record, "TREATMENT_NAME")
    return [i["Text"].lower() for i in candidates]


def extract_prescriptions(record):
    candidates = _extract_entity_category(record, "MEDICATION")
    parsed_medications = []
    for candidate in candidates:
        output = {"drug": candidate["Text"]}
        if "Attributes" not in candidate:
            parsed_medications.append(output)
            continue
        candidate_dict = {i["Type"]: i for i in candidate["Attributes"]}
        if "ROUTE_OR_MODE" in candidate_dict:
            output["route"] = candidate_dict["ROUTE_OR_MODE"]["Text"]
        if "STRENGTH" in candidate_dict:
            output["dose"] = candidate_dict["STRENGTH"]["Text"]
        if "FORM" in candidate_dict:
            output["form"] = candidate_dict["FORM"]["Text"]
        parsed_medications.append(output)
    return parsed_medications


def extract_provider(record):
    candidates = _extract_entity_type(record, "NAME")
    if len(candidates) > 1:
        # If more than one name is mentioned return the less frequently mentioned name
        return _sorted_mention(candidates, "Text")[-1]
    candidates = _extract_entity_type(record, "ADDRESS")
    for candidate in candidates:
        for likely_text in [
            "emergency room",
            "ward",
        ]:
            if likely_text in candidate["Text"].lower():
                return candidate["Text"]
    return None


DEFAULT_IDENTIFIERS = {
    "nhs_number": extract_nhs_number,
    "name": extract_name,
    "date_of_birth": extract_date_of_birth,
    "disease": extract_disease,
    "symptoms": extract_symptoms,
    "date_of_visit": extract_date_of_visit,
    "department": extract_department,
    "gender": extract_gender,
    "ethnicity": extract_ethnicity,
    "treatment": extract_treatment,
    "prescriptions": extract_prescriptions,
    "provider": extract_provider,
}


def extract_identifiers(identifier_extractors, record):
    """Extracts identifiers from a record
    Uses the provided dictionary of identifier names and extractors to extract identifiers from
    structured records.
    :param identifier_extractors: Dictionary of identifier names and functions to extract them.
    :param record: Output from the NER.
    :returns: A structured record"""
    processed_record = {
        name: extractor(record["Entities"])
        for name, extractor in identifier_extractors.items()
    }
    return Record(**processed_record)


def prepare_common_records(
    identifier_extractors: Dict[str, Callable[[Dict], Any]],
    records: List[Dict],
) -> List[Record]:
    """Extracts identifiers for all records in a dataset
    Uses the provided dictionary of identifier names and extractors to extract identifiers.
    :param identifier_extractors: Dictionary of identifier names and functions to extract them.
    :param records: Output from the NER.
    :returns: List of structured records"""
    return [
        extract_identifiers(identifier_extractors, record)
        for record in records
    ]


def calculate_ner_cost(
    records: Union[pd.DataFrame, List], column_name: Optional[str] = None
) -> float:
    """Calculating cost in US dollars based on March 2023 pricing for the Amazon Comprehend Medical
    NER service"""
    if isinstance(records, pd.DataFrame):
        tokens = records[column_name].str.len().sum() / 100
    elif isinstance(records, list):
        tokens = sum([len(i) for i in records]) / 100
    else:
        raise TypeError(
            "calculate_ner_costs accepts either a DataFrame or list"
        )
    cost = tokens * 0.01

    return cost
