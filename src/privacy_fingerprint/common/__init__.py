from typing import Dict, List, Optional, Tuple

import pandas as pd
from fuzzywuzzy import fuzz, process
from pydantic import BaseModel


class Prescription(BaseModel):
    drug: str
    dose: Optional[str]
    route: Optional[str]
    form: Optional[str]


class Record(BaseModel):
    """Data object holding structured clinical information"""

    nhs_number: Optional[str]
    name: Optional[str]
    date_of_birth: Optional[str]
    disease: Optional[List[str]]
    symptoms: List[str]
    date_of_visit: Optional[str]
    department: Optional[str]
    gender: Optional[str]
    ethnicity: Optional[str]
    treatment: List[str]
    prescriptions: Optional[List[Prescription]]
    provider: Optional[str]


def _compare_text(in_record: str, out_record: str) -> Tuple[float, int, float]:
    if out_record is None:
        summary = 0
    else:
        summary = fuzz.ratio(in_record, out_record)
    score = summary / 100
    return score, 1, summary


def _compare_date(in_record: str, out_record: str) -> Tuple[float, int, float]:
    out_date = pd.to_datetime(out_record, errors="coerce")
    if pd.isnull(out_date):
        summary = 0
    else:
        in_date_str = str(pd.to_datetime(in_record, errors="coerce").date())
        out_date_str = str(pd.to_datetime(out_record, errors="coerce").date())
        summary = fuzz.ratio(in_date_str, out_date_str)
    score = summary / 100
    return score, 1, summary


def _compare_list(
    in_record: List[str], out_record: List[str]
) -> Tuple[float, int, float]:
    max_score = 0
    score = 0
    if (in_record is None) or (len(in_record) == 0):
        return 0, 0, None
    if len(out_record) > 0:
        for in_field in in_record:
            max_score += 1
            best_match = process.extractOne(
                in_field.lower(), [i.lower() for i in out_record]
            )
            score += best_match[1] / 100
    summary = 0 if (max_score == 0) else (100 * score / max_score)
    return score, max_score, summary


def compare_common_records(
    in_record: Record, out_record: Record
) -> Tuple[float, int, Dict[str, float]]:
    """Compare two records for agreement

    :param in_record: The base record to compare against, e.g. from synthea
    :param out_record: The record to compare, e.g. the record after LLM and NER
    :returns: Total score, total possible score and a field-by-field breakdown
    """
    overall_score = 0
    max_score = 0
    summary = {}
    # NHS number
    score, field_max_score, field_summary = _compare_text(
        in_record.nhs_number, out_record.nhs_number
    )
    overall_score += score
    max_score += field_max_score
    summary["nhs_number"] = field_summary
    # Name
    score, field_max_score, field_summary = _compare_text(
        in_record.name, out_record.name
    )
    overall_score += score
    max_score += field_max_score
    summary["name"] = field_summary
    # Date of birth
    score, field_max_score, field_summary = _compare_date(
        in_record.date_of_birth, out_record.date_of_birth
    )
    overall_score += score
    max_score += field_max_score
    summary["date_of_birth"] = field_summary
    # Disease
    score, field_max_score, field_summary = _compare_list(
        in_record.disease, out_record.disease
    )
    overall_score += score
    max_score += field_max_score
    summary["disease"] = field_summary
    # Date of visit
    score, field_max_score, field_summary = _compare_date(
        in_record.date_of_visit, out_record.date_of_visit
    )
    overall_score += score
    max_score += field_max_score
    summary["date_of_visit"] = field_summary
    # Department
    score, field_max_score, field_summary = _compare_text(
        in_record.department, out_record.department
    )
    overall_score += score
    max_score += field_max_score
    summary["department"] = field_summary
    # Gender
    max_score += 1
    if out_record.gender is None:
        summary["gender"] = 0
    else:
        summary["gender"] = 100 * (in_record.gender == out_record.gender)
    overall_score += summary["gender"] / 100
    # Ethnicity
    score, field_max_score, field_summary = _compare_text(
        in_record.ethnicity, out_record.ethnicity
    )
    overall_score += score
    max_score += field_max_score
    summary["ethnicity"] = field_summary
    # Treatment
    score, field_max_score, field_summary = _compare_list(
        in_record.treatment, out_record.treatment
    )
    overall_score += score
    max_score += field_max_score
    summary["treatment"] = field_summary
    # Prescriptions - only looking at the drug name, not dose, form or route
    max_prescription = 0
    overall_prescription = 0
    if (in_record.prescriptions is not None) and (
        len(out_record.prescriptions) > 0
    ):
        for in_prescription in in_record.prescriptions:
            max_prescription += 1
            best_match = process.extractOne(
                in_prescription.drug,
                [i.drug for i in out_record.prescriptions],
            )
            overall_prescription += best_match[1] / 100
    max_score += max_prescription
    overall_score += overall_prescription
    if in_record.prescriptions is None:
        summary["prescription"] = None
    elif max_prescription == 0:
        summary["prescription"] = 0
    else:
        summary["prescription"] = 100 * overall_prescription / max_prescription
    # Provider
    score, field_max_score, field_summary = _compare_text(
        in_record.provider, out_record.provider
    )
    overall_score += score
    max_score += field_max_score
    summary["provider"] = field_summary
    return overall_score, max_score, summary
