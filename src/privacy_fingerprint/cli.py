import json
import logging
import os
import sys
from typing import List, Tuple

import click
import pandas as pd
import yaml

from privacy_fingerprint.common import Record
from privacy_fingerprint.common.config import (
    load_experiment_config,
    load_experiment_config_from_file,
    load_global_config_from_file,
)

from privacy_fingerprint.explain import PrivacyRiskExplainer
from privacy_fingerprint.extract.aws_comprehend import (
    DEFAULT_IDENTIFIERS,
    ComprehendExtractor,
    calculate_ner_cost,
    prepare_common_records,
)
from privacy_fingerprint.score import PrivacyRiskScorer, encode, preprocess


def prepare_results_directory(results_directory: str):
    """Create the output directory, copy config and set up logging to file

    :param results_directory: Directory location"""
    try:
        os.mkdir(results_directory)
    except FileExistsError:
        logging.error("The results directory already exists")
        sys.exit(2)
    except FileNotFoundError:
        logging.error(
            "The parent directory for the results directory does not exist"
        )
        sys.exit(2)

    fh = logging.FileHandler(os.path.join(results_directory, "experiment.log"))
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    with open(
        os.path.join(results_directory, "experiment_config.yaml"), "w"
    ) as fp:
        yaml.safe_dump(load_experiment_config().dict(), fp)


def load_data(data_path: str) -> Tuple[pd.DataFrame, str, pd.Index]:
    """Load the data for processing

    :param data_path: Data file location
    :returns: Data frame, column name containing data, and the index"""
    records = pd.read_csv(data_path)
    column_names = [i for i in records.columns if i != "Unnamed: 0"]
    if len(column_names) > 1:
        logging.error(
            "Data file contains unexpected columns."
            " A single column with an optional unnamed index is expected"
        )
        sys.exit(2)
    index = records.index
    if "Unnamed: 0" in records.columns:
        index = records["Unnamed: 0"]
    column_name = column_names[0]
    logging.info(f"Processing column {column_name}")
    return records, column_name, index


def prepare_standardised_records(
    records: pd.DataFrame, column_name: str, results_directory: str
) -> List[Record]:
    """Process free text to generate a set of standardised records

    :param records: Data frame containing the free text notes
    :param column_name: Column in the data frame containing the data
    :param results_directory: Directory to save data
    :returns: Standardised list of records
    """
    logging.info("Extracting information from text.")
    extractor = ComprehendExtractor()
    extracted_records = [
        extractor.extract_record(record)
        for record in records[column_name].tolist()
    ]

    with open(
        os.path.join(results_directory, "NER_raw_extract.json"), "w"
    ) as fp:
        json.dump(extracted_records, fp)

    standardised_records = prepare_common_records(
        DEFAULT_IDENTIFIERS, extracted_records
    )

    with open(
        os.path.join(results_directory, "NER_processed_extract.json"), "w"
    ) as fp:
        json.dump([i.dict() for i in standardised_records], fp)
    return standardised_records


def calculate_privacy_score(
    standardised_records: List[Record], index: pd.Index, results_directory: str
) -> Tuple[float, pd.DataFrame, pd.DataFrame, PrivacyRiskScorer]:
    """Calculate the global and individual privacy risk scores

    :param standardised_records: List of records in a standardised format
    :param index: The original dataset index
    :param results_directory: Directory to save data
    :returns: The overall population uniqueness, an encoded version of the
    dataset, individual record privacy risk scores, and the scoring object"""
    logging.info("Preparing data for privacy risk scoring.")
    df_records = preprocess(standardised_records)
    df_records.index = index.values

    df_records.to_csv(os.path.join(results_directory, "scoring_dataset.csv"))

    encoded_dataset, lookup = encode(df_records)
    logging.info("Calculating risk score.")
    logging.info("Scoring can take several minutes")
    scorer = PrivacyRiskScorer()
    population_score = scorer.calculate_population_uniqueness(encoded_dataset)
    logging.info(f"Population uniqueness is {population_score}")
    scorer.fit(encoded_dataset)
    individual_scores = scorer.predict(encoded_dataset)
    individual_scores.to_csv(
        os.path.join(results_directory, "individual_scores.csv")
    )
    return population_score, encoded_dataset, individual_scores, scorer


def explain_privacy_scores(
    encoded_dataset: pd.DataFrame,
    scorer: PrivacyRiskScorer,
    results_directory: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Calculate the contribution of individual identifiers to the overall
    privacy risk scores

    :param encoded_dataset: The dataset after pre-processing and encoding
    :param scorer: The privacy scoring object
    :returns: The individual record explainability and the global identifier
    contributions
    """
    logging.info("Calculating contribution from individual identifiers")
    logging.info("This step can take hours depending on the dataset")
    transformed_dataset = scorer.map_records_to_copula(encoded_dataset)
    explainer = PrivacyRiskExplainer(
        scorer.predict_transformed, transformed_dataset.shape[1]
    )
    explanations, global_explanation, shap_explain = explainer.explain(
        transformed_dataset
    )
    explanations.to_csv(os.path.join(results_directory, "explanations.csv"))
    global_explanation.to_csv(
        os.path.join(results_directory, "global_explanation.csv")
    )
    return explanations, global_explanation, shap_explain


@click.group()
@click.argument("global_config")
def cli(global_config):
    """Interact with the privacy fingerprint risk scoring package

    GLOBAL_CONFIG contains configuration information shared across multiple
    runs
    """
    load_global_config_from_file(global_config)


@click.command()
@click.argument("experiment_config")
@click.argument("data_path")
@click.argument("results_directory")
@click.option(
    "--explain/--no-explain",
    default=False,
    help="Explain privacy risk scores using SHAP",
)
def assess(experiment_config, data_path, results_directory, explain):
    """Display the privacy risk for the dataset contained in DATA_PATH.

    EXPERIMENT_CONFIG contains configuration information that is specific to a single run of the tool

    DATA_PATH is the data to be analysed and should be a csv file with a single column

    RESULTS_DIRECTORY will be used to save all results"""
    # Set up logging to the screen and file
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    load_experiment_config_from_file(experiment_config)
    records, column_name, index = load_data(data_path)

    prepare_results_directory(results_directory)

    log_ner_cost(column_name, records)

    standardised_records = prepare_standardised_records(
        records, column_name, results_directory
    )

    (
        population_score,
        encoded_dataset,
        individual_scores,
        scorer,
    ) = calculate_privacy_score(standardised_records, index, results_directory)

    if explain:
        (
            explanations,
            global_explanation,
            shap_explain,
        ) = explain_privacy_scores(encoded_dataset, scorer, results_directory)
    logging.info(f"Results of this run can be found in {results_directory}")


def log_ner_cost(column_name, records):
    cost = calculate_ner_cost(records, column_name)
    logging.info(
        (
            "Calculating cost based on March 2023 pricing for the Amazon "
            "Comprehend Medical NER service"
        )
    )
    logging.info(
        (
            "This estimate may be incorrect if pricing has changed or if the "
            "cache can be used to avoid API calls"
        )
    )
    logging.info(f"Estimated cost is ${cost:.2f}")


cli.add_command(assess)
