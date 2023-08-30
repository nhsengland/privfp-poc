import copy
import os
import random
import re
import shlex
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from privacy_fingerprint.common import Record
from privacy_fingerprint.common.config import (
    load_experiment_config,
    load_global_config,
)


def generate_records(output_directory: str) -> List[Dict[str, Any]]:
    """Generate records using Synthea.

    Records are currently generated for emergency room admissions in Hampshire.

    Run a subprocess to generate records outside of Python and persist them to
    a given directory.

    :param output_directory: Output records to this directory
    :returns: A list of records"""
    global_config = load_global_config()
    experiment_config = load_experiment_config()
    dummy_data, export_directory, logs = run_synthea(
        experiment_config.synthea.num_records,
        experiment_config.synthea.county,
        global_config.synthea.install_directory,
        output_directory,
        experiment_config.synthea.extra_config,
    )
    return prepare_records(
        dummy_data, encounter_type=experiment_config.synthea.encounter_type
    )


def create_synthea_command(
    patient_number: int,
    county: str,
    export_directory: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    extra_config = ""
    if config is not None:
        extra_config = " ".join([f"--{k}={v}" for k, v in config.items()])
    cmd_str = (
        f"./run_synthea --exporter.baseDirectory={export_directory} "
        f"{extra_config} -p {patient_number} {county}"
    )
    cmd = shlex.split(cmd_str)
    return cmd


def run_synthea_process(
    command: List[str],
    synthea_directory: str,
) -> List[str]:
    logs = []
    with subprocess.Popen(
        command,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        bufsize=1,
        cwd=synthea_directory,
        encoding="utf-8",
    ) as proc:
        for line in iter(proc.stdout.readline, ""):
            logs.append(line)
    return logs


def load_dummy_data_from_directory(
    directory_path: str,
) -> Dict[str, pd.DataFrame]:
    """Load dummy data from the export directory of Synthea

    :param directory_path Export directory used by Synthea
    :returns: Dictionary containing dummy data for:
        - Patients
        - Encounters
        - Observations
        - Conditions
        - Procedures
        - Care plans
        - Immunizations
        - Organizations
        - Providers
    """
    experiment_config = load_experiment_config()
    output = {
        i: pd.read_csv(os.path.join(directory_path, "csv", f"{i}.csv"))
        for i in ["patients", "encounters", "organizations", "providers"]
    }
    if experiment_config.synthea.encounter_type is not None:
        if isinstance(experiment_config.synthea.encounter_type, list):
            desc_filter = experiment_config.synthea.encounter_type
        else:
            desc_filter = [
                experiment_config.synthea.encounter_type,
            ]
        encounter_ids = (
            output["encounters"]
            .loc[output["encounters"].DESCRIPTION.isin(desc_filter)]
            .Id.tolist()
        )
        output["encounters"] = output["encounters"].loc[
            output["encounters"].Id.isin(encounter_ids)
        ]
    else:
        encounter_ids = None
    for i in [
        "observations",
        "medications",
        "conditions",
        "procedures",
        "careplans",
        "immunizations",
    ]:
        dataset = pd.read_csv(os.path.join(directory_path, "csv", f"{i}.csv"))
        if encounter_ids is not None:
            dataset = dataset.loc[dataset.ENCOUNTER.isin(encounter_ids)].copy()
        output[i] = dataset
    return output


def display_logs(txt, line_len=80):
    output = []
    txt = txt.split("\n")
    for t in txt:
        if len(t) > line_len:
            output.extend(
                [
                    t[i : i + line_len]  # noqa: E203
                    for i in range(0, len(t), line_len)
                ]
            )
            continue
        output.append(t)
    for line in output:
        print(line)


def run_synthea(
    patient_number: int,
    county: str,
    synthea_directory: str,
    export_directory: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], str, List[str]]:
    """Generate dummy data using Synthea

    :param patient_number Number of dummy patients to generate. Additional patients may
    be generated that have passed away.
    :param county The geographic county from which dummy patient records are created
    :param synthea_directory The directory in which Synthea is available
    :param export_directory The directory in which to store the generated data.
    :param config Additional configuration options to be passed to Synthea
    :returns: A tuple of:
        Dictionary containing dummy data for:
            - Patients
            - Encounters
            - Observations
            - Conditions
            - Procedures
            - Care plans
            - Immunizations
            - Organizations
            - Providers
        Directory containing dummy data
        Logs produced by Synthea
    """
    abs_export_directory = os.path.abspath(export_directory)
    cmd = create_synthea_command(
        patient_number, county, abs_export_directory, config
    )
    logs = run_synthea_process(cmd, synthea_directory)
    if (
        "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.\n"
        in logs
    ):
        raise Exception(
            "JAVA_HOME is not set and java command not found on path"
        )
    dummy_data = load_dummy_data_from_directory(export_directory)
    return dummy_data, abs_export_directory, logs


def generate_full_name(patient_record: pd.Series) -> str:
    """Generate a complete name in text from the multiple fields

    Synthea creates names like Buddy123 Bernier987 so digits are removed and
    parts (title, first, family) combined

    :param patient_record: A single patient row as generated by Synthea
    :returns: The patient name including title, first, family, and any suffix
    """
    full_name = "{0} {1}".format(
        re.sub(r"\d", "", patient_record.FIRST),
        re.sub(r"\d", "", patient_record.LAST),
    )
    if not pd.isnull(patient_record.PREFIX):
        full_name = f"{patient_record.PREFIX} {full_name}"
    if not pd.isnull(patient_record.SUFFIX):
        full_name = f"{full_name} {patient_record.SUFFIX}"
    return full_name


def process_provider_name(provider_record: pd.Series) -> str:
    """Generate a name from a provider record

    Synthea creates names like Buddy123 Bernier987 so digits are removed

    :param provider_record: A single provider row as generated by Synthea
    :returns: The provider name with digits removed"""
    full_name = re.sub(r"\d", "", provider_record.NAME)
    return full_name


def expand_postcode(postcode: str) -> str:
    """Add the in-code to a partial postcode

    :param postcode: The partial postcode as supplied by Synthea
    :returns: An expanded postcode with the second part randomly generated
    """
    letters = "ABDEFGHJLNPQRSTUWXYZ"
    return f"{postcode} {random.randint(0,9)}{random.choice(letters)}{random.choice(letters)}"


def generate_nhs_number(patient_record: pd.Series) -> str:
    """Generate a NHS number from Synthea generated data

    :param patient_record: A single patient row as generated by Synthea
    :returns: A 10 digit number to simulate an NHS number
    """
    nhs_num = patient_record.SSN.replace("-", "")
    if pd.isnull(patient_record.PASSPORT):
        start_digits = nhs_num[0:3]
    else:
        start_digits = patient_record.PASSPORT[1:4]
    check = random.randint(0, 9)
    return f"{start_digits} {nhs_num[3:6]} {nhs_num[6:]}{check}"


def parse_patient_details(patient_record: pd.Series) -> Dict:
    """Parse the details for a patient

    :param patient_record: A single patient row as generated by Synthea
    :returns: A modified patient record with values more appropriate for the UK
    """
    gender_lookup = {"M": "male", "F": "female"}
    marital_lookup = {
        "M": "married",
        "S": "single",
    }
    experiment_config = load_experiment_config()
    ethnicity_lookup = experiment_config.synthea.ethnicity_types
    output = {
        "name": generate_full_name(patient_record),
        "NHS number": generate_nhs_number(patient_record),
        "address": f"{patient_record.ADDRESS}, {patient_record.CITY}, {expand_postcode(patient_record.ZIP)}",
        "date of birth": patient_record.BIRTHDATE,
        "marital status": marital_lookup.get(
            patient_record.MARITAL, patient_record.MARITAL
        ),
        "ethnicity": random.choice(ethnicity_lookup),
        "gender": gender_lookup.get(
            patient_record.GENDER, patient_record.GENDER
        ),
    }
    return output


def get_encounters_for_patient(
    encounters: pd.DataFrame,
    patient_id: str,
    descriptions: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """Get medical encounters for a specific patient, optionally filtered for encounter type

    :param encounters: A dataframe of encounters as generated from Synthea
    :param patient_id: ID for a single patient in the Synthea dataset
    :param descriptions: The types of encounter to return
    :returns: A dataframe with encounters for the patient
    """
    if descriptions is not None:
        if isinstance(descriptions, list):
            desc_filter = descriptions
        else:
            desc_filter = [
                descriptions,
            ]
        return encounters.loc[
            (encounters.PATIENT == patient_id)
            & (encounters.DESCRIPTION.isin(desc_filter))  # noqa: W503
        ]
    return encounters.loc[encounters.PATIENT == patient_id]


def get_records_for_encounter(
    records: pd.DataFrame, encounter_id: str
) -> pd.DataFrame:
    """Get records associated with an event from the observations, medications, etc records

    :param records: Dataframe containing records generated from Synthea
    :param encounter_id: An encounter ID from the Synthea dataset
    :returns: A dataframe containing all records related to encounter_id"""
    encounter_records = records.loc[records.ENCOUNTER == encounter_id]
    if "START" in encounter_records.columns:
        return encounter_records.sort_values("START")
    if "DATE" in encounter_records.columns:
        return encounter_records.sort_values("DATE")
    return encounter_records


def parse_provider(
    providers: pd.DataFrame, organizations: pd.DataFrame, encounter: pd.Series
) -> Dict:
    """Construct a record for the medical provider in an encounter

    :param providers: Dataframe of providers as generated by Synthea
    :param organizations: Dataframe of organizations as generated by Synthea
    :param encounter: A single encounter record as generated by Synthea
    :returns: Description of the provider and organization where the encounter happened
    """
    e_org = organizations.loc[organizations.Id == encounter.ORGANIZATION].iloc[
        0
    ]
    e_provider = providers.loc[providers.Id == encounter.PROVIDER].iloc[0]
    return {
        "doctor": process_provider_name(e_provider),
        "facility": e_org.NAME,
    }


def parse_encounter(
    encounter_id: str,
    encounters: pd.DataFrame,
    observations: pd.DataFrame,
    medications: pd.DataFrame,
    conditions: pd.DataFrame,
    procedures: pd.DataFrame,
    immunizations: pd.DataFrame,
    careplans: pd.DataFrame,
    organizations: pd.DataFrame,
    providers: pd.DataFrame,
) -> Dict:
    """Construct a record suitable for text generation for an encounter

    :param encounter_id The ID for the encounter
    :param encounters The collection of encounters for the dummy dataset
    :param observations The collection of observations for the dummy dataset
    :param medications The collection of medications for the dummy dataset
    :param conditions The collection of conditions for the dummy dataset
    :param procedures The collection of procedures for the dummy dataset
    :param immunizations The collection of immunizations for the dummy dataset
    :param careplans The collection of careplans for the dummy dataset
    :param organizations The collection of organizations for the dummy dataset
    :param providers The collection of providers for the dummy dataset
    :returns: Complete record for an encounter
    """
    encounter_observations = get_records_for_encounter(
        observations, encounter_id
    )
    encounter_medications = get_records_for_encounter(
        medications, encounter_id
    )
    encounter_conditions = get_records_for_encounter(conditions, encounter_id)
    encounter_procedures = get_records_for_encounter(procedures, encounter_id)
    encounter_careplans = get_records_for_encounter(careplans, encounter_id)
    encounter = encounters.loc[encounters.Id == encounter_id].iloc[0]
    output = {
        "visit type": encounter.DESCRIPTION,
        "visit date": encounter.START,
        "provider": parse_provider(providers, organizations, encounter),
        "visit reason": _multiple_replace(encounter.REASONDESCRIPTION),
    }
    if not encounter_medications.DESCRIPTION.empty:
        output[
            "prescription"
        ] = encounter_medications.DESCRIPTION.unique().tolist()
    if not encounter_observations.DESCRIPTION.empty:
        output["observations"] = [
            {k.lower(): v[k] for k in ["DESCRIPTION", "VALUE", "UNITS"]}
            for _, v in encounter_observations.iterrows()
        ]
    if not encounter_conditions.DESCRIPTION.empty:
        output["conditions"] = [
            _multiple_replace(i)
            for i in encounter_conditions.DESCRIPTION.unique().tolist()
        ]
    if not encounter_procedures.DESCRIPTION.empty:
        output["procedures"] = [
            {
                k.lower(): _multiple_replace(v[k])
                for k in ["DESCRIPTION", "REASONDESCRIPTION"]
            }
            for _, v in encounter_procedures.iterrows()
        ]
    if not encounter_careplans.DESCRIPTION.empty:
        output["care plan"] = [
            {
                k.lower(): _multiple_replace(v[k])
                for k in ["DESCRIPTION", "REASONDESCRIPTION"]
            }
            for _, v in encounter_careplans.iterrows()
        ]
    return output


def _multiple_replace(
    text, replacements=["(disorder)", "(record artifact)", "(procedure)"]
):
    """Replace multiple strings in a piece of text

    Synthea adds unnecssary text on the end of diseases etc that
    this function can remove."""
    if pd.isnull(text):
        return text
    for replacement in replacements:
        text = text.replace(replacement, "")
    return text.strip()


def prepare_records(
    dummy_data: Dict[str, pd.DataFrame],
    encounter_type: str = "Emergency room admission (procedure)",
    records_per_patient: int = 1,
) -> List[Dict[str, Any]]:
    """
    Parse records as generated by Synthea

    :param dummy_data: The data generated by Synthea
    :param encounter_type: The type of encounter to process
    :param records_per_patient: The number of records to randomly select for each patient
    :returns: The parsed set of records
    """
    experiment_config = load_experiment_config()
    patients = dummy_data["patients"]
    encounters = dummy_data["encounters"]
    observations = dummy_data["observations"]
    medications = dummy_data["medications"]
    conditions = dummy_data["conditions"]
    procedures = dummy_data["procedures"]
    careplans = dummy_data["careplans"]
    immunizations = dummy_data["immunizations"]
    organizations = dummy_data["organizations"]
    providers = dummy_data["providers"]
    processed_patient_records = []

    for _, row in patients.iterrows():
        patient_details = parse_patient_details(row)
        selected_records = get_encounters_for_patient(
            encounters, row.Id, [encounter_type]
        )
        if selected_records.empty:
            continue
        for _, encounter in selected_records.sample(
            experiment_config.synthea.records_per_patient
        ).iterrows():
            encounter_id = encounter.Id
            encounter_details = copy.deepcopy(patient_details)
            encounter_details.update(
                parse_encounter(
                    encounter_id,
                    encounters,
                    observations,
                    medications,
                    conditions,
                    procedures,
                    immunizations,
                    careplans,
                    organizations,
                    providers,
                )
            )
            processed_patient_records.append(encounter_details)
    return processed_patient_records


def extract_nhs_number(record):
    return record["NHS number"]


def extract_name(record):
    return record["name"]


def extract_date_of_birth(record):
    return record["date of birth"]


def extract_disease(record):
    return record.get("conditions", None)


def extract_symptoms(record):
    if pd.isnull(record["visit reason"]):
        return record.get("conditions", [])
    else:
        return [
            record["visit reason"],
        ]


def extract_date_of_visit(record):
    return record["visit date"]


def extract_department(record):
    return record["provider"]["facility"]


def extract_gender(record):
    return record["gender"]


def extract_ethnicity(record):
    return record["ethnicity"]


def extract_treatment(record):
    return [
        i["description"]
        for i in record.get("procedures", []) + record.get("care plan", [])
    ]


def extract_prescriptions(record):
    prescriptions = record.get("prescription", None)
    if prescriptions is None:
        return None
    parsed_prescriptions = []
    for prescription in prescriptions:
        m = re.match(r"^(.+) (\d+ .{2,3})(\s.*)? (.+)$", prescription)
        if m is None:
            continue
        parsed_prescriptions.append(
            {
                "drug": m.group(1),
                "dose": m.group(2),
                "route": m.group(3),
                "form": m.group(4),
            }
        )
    if len(parsed_prescriptions) == 0:
        return None
    return parsed_prescriptions


def extract_provider(record):
    return record["provider"]["doctor"]


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
    Uses the provided dictionary of identifier names and extractors to extract identifiers
    from structured records.
    :param identifier_extractors: Dictionary of identifier names and functions to extract them.
    :param record: Structured record as generated by Synthea.
    :returns: A structured record"""
    processed_record = {
        name: extractor(record)
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
    :param records: Structured record as generated by Synthea.
    :returns: List of structured records"""
    return [
        extract_identifiers(identifier_extractors, record)
        for record in records
    ]
