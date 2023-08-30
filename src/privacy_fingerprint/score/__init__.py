from collections import defaultdict
from typing import Dict, List, Tuple

import julia
import numpy as np
import pandas as pd
from julia.api import Julia
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from privacy_fingerprint.common import Record
from privacy_fingerprint.common.config import (
    load_experiment_config,
    load_global_config,
)


def preprocess(records: List[Record]) -> pd.DataFrame:
    """Prepare records for pycorrectmatch

    This includes
    * limiting the information to an agreed set of identifiers
    * checking types and encoding
    * selecting individual values from columns that contain multiple entries (e.g. co-morbidities)

    :params records: List of records in a standardised format
    :returns: DataFrame ready for analysis by pycorrectmatch"""
    config = load_experiment_config()
    if config.scoring.encoding_scheme == "one-hot":
        all_output = _one_hot_preprocess(records)
    elif config.scoring.encoding_scheme == "rarest":
        all_output = _rarest_preprocess(records)
    else:
        raise Exception(
            f"Encoding scheme ({config.scoring.encoding_scheme}) is not recognised"
        )
    all_output = pd.DataFrame(all_output)
    return all_output


def _generate_basic_columns(record: Record) -> Dict:
    """Prepare columns common to all encoding schemes

    :param record: A record in standardised format
    :returns: Dictionary of values"""
    output = {
        "nhs_number": record.nhs_number,
        "name": record.name,
        "date_of_birth": record.date_of_birth,
        "date_of_visit": record.date_of_visit,
        "department": record.department,
        "gender": record.gender,
        "ethnicity": record.ethnicity,
        "provider": record.provider,
    }
    return output


def _one_hot_preprocess(records: List[Record]) -> List[Dict]:
    """Prepare records using a one-hot encoding scheme

    :param records: List of records in a standardised format
    :returns: DataFrame ready for analysis by pycorrectmatch"""
    config = load_experiment_config()
    one_hot_columns = defaultdict(int)
    for record in records:
        if record.disease is not None:
            for disease in record.disease:
                one_hot_columns[("disease", disease)] += 1
        for symptom in record.symptoms:
            one_hot_columns[("symptoms", symptom)] += 1
        for treatment in record.treatment:
            one_hot_columns[("treatment", treatment)] += 1
        if record.prescriptions is not None:
            for prescription in record.prescriptions:
                one_hot_columns[("prescriptions", prescription.drug)] += 1
    num_cols = 8
    remaining_cols = config.scoring.max_columns - num_cols
    selected_columns = sorted(one_hot_columns.items(), key=lambda x: x[1])[
        :remaining_cols
    ]
    all_output = []
    for record in records:
        output = _generate_basic_columns(record)
        for (col_type, col_detail), _ in selected_columns:
            if record.__getattribute__(col_type) is None:
                continue
            col_label = f"{col_type}_{col_detail.replace(' ', '_')}"
            output[col_label] = 0
            if col_type == "prescriptions":
                for prescription in record.prescriptions:
                    if prescription.drug == col_detail:
                        output[col_label] = prescription.dose
            else:
                if col_detail in record.__getattribute__(col_type):
                    output[col_label] = 1
        all_output.append(output)
    return all_output


def _rarest_preprocess(records: List[Record]) -> List[Dict]:
    """Prepare records using the rarest disease etc mentioned

    :param records: List of records in a standardised format
    :returns: DataFrame ready for analysis by pycorrectmatch"""
    one_hot_columns = defaultdict(int)
    for record in records:
        if record.disease is not None:
            for disease in record.disease:
                one_hot_columns[("disease", disease)] += 1
        for symptom in record.symptoms:
            one_hot_columns[("symptoms", symptom)] += 1
        for treatment in record.treatment:
            one_hot_columns[("treatment", treatment)] += 1
        if record.prescriptions is not None:
            for prescription in record.prescriptions:
                one_hot_columns[("prescriptions", prescription.drug)] += 1
    all_output = []
    for record in records:
        output = _generate_basic_columns(record)
        for col_type in ["disease", "symptoms", "treatment", "prescriptions"]:
            if record.__getattribute__(col_type) is None:
                continue
            if len(record.__getattribute__(col_type)) == 0:
                output[col_type] = ""
                continue
            sorted_col = [
                m
                for m, n in sorted(
                    [
                        (j, k)
                        for (i, j), k in one_hot_columns.items()
                        if i == col_type
                    ],
                    key=lambda x: x[1],
                )
            ]
            if col_type == "prescriptions":
                detail_key = min(
                    [
                        sorted_col.index(i.drug)
                        for i in record.__getattribute__(col_type)
                    ]
                )
            else:
                detail_key = min(
                    [
                        sorted_col.index(i)
                        for i in record.__getattribute__(col_type)
                    ]
                )
            output[col_type] = sorted_col[detail_key]
        all_output.append(output)
    return all_output


def encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Encode a dataset for processing by pycorrectmatch

    :param df: Table of data
    :returns: Table of encoded data and lookup dictionary"""
    output = df.fillna(0)
    for col in output.columns:
        if len(set([type(i) for i in output[col].tolist()])) > 1:
            output[col] = output[col].astype("str")
    ct = ColumnTransformer(
        [
            (f"encode_{col_name}", PipelineLabelEncoder(), col_name)
            for col_name in output.columns
        ]
    )
    encoded_output = pd.DataFrame(
        ct.fit_transform(output), index=output.index, columns=output.columns
    )
    transformers = {k: j for i, j, k in ct.transformers_}
    lookup = {
        i: {k: j for j, k in enumerate(transformers[i].classes_)}
        for i in output.columns
    }
    return encoded_output, lookup


class PipelineLabelEncoder(LabelEncoder):
    def fit_transform(self, y, _, *args, **kwargs):
        return super().fit_transform(y, *args, **kwargs).reshape(-1, 1)

    def transform(self, y, _, *args, **kwargs):
        return super().transform(y, *args, **kwargs).reshape(-1, 1)


class PrivacyRiskResults:
    """A wrapper for the output of pycorrectmatch"""

    pass


class PrivacyRiskScorer:
    """A scorer using PyCorrectMatch that takes a pd.DataFrame and calculates
    population uniqueness, fits a Gaussian copula and calculates individual uniqueness scores."""

    def __init__(self):
        config = load_global_config()

        julia.install(julia=config.pcm.path_to_julia)
        Julia(compiled_modules=False, runtime=config.pcm.path_to_julia)
        import correctmatch  # noqa E402

        correctmatch.precompile()

        self.correctmatch = correctmatch
        self.fitted_model = None
        self.size = 0

    def calculate_population_uniqueness(self, df: pd.DataFrame) -> float:
        return self.correctmatch.uniqueness(df.values)

    def fit(self, df):
        """
        Fit a Gaussian copula model to a discrete multivariate dataset
        The fitted model is a Julia object with the estimated model.
        The argument `exact_marginal`=True ensures that the marginal distributions
        are categorical, whose values range from 1 to number_of_unique_values for each feature.
        """
        self.size = df.shape[0]
        self.look_up = self.create_lookup_dict(df)
        self.fitted_model = self.correctmatch.fit_model(
            df.values, exact_marginal=True
        )

    def create_lookup_dict(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[int, int]]:
        """
        Create a dictionary with one key per column and another dictionary
        as a value. That inner dictionary has the mapping between the value of the
        record and the marginal distribution value it corresponds to for a fitted copula.

        :param df: pd.DataFrame to be scored
        :returns Dictionary that maps real feature values to the values of the marginal
        distributions of the copula model.
        """
        record_to_copula = {
            col: dict(
                zip(
                    df[col].value_counts().index,
                    range(1, df[col].nunique() + 1),
                )
            )
            for col in df.columns.to_list()
        }
        return record_to_copula

    def map_record_to_copula(self, row: pd.Series) -> List[int]:
        """Convert real dataset values of a record to values that correspond to
        the marginal distributions of the fitted copula using the look up dictionary.

        :param row: pd.Series to be transformed from real values to the values of the
        marginal distributions of the copula model
        :returns List of integer values, the transformed record
        """
        if self.fitted_model is None:
            raise Exception("Please fit the model first.")
        return [self.look_up[col][row.loc[col]] for col in row.index]

    def get_individual_uniqueness(self, row: pd.Series) -> float:
        """Estimate individual uniquess for one given record.

        :param row: pd.Series to be transformed from real values to the values of the
        marginal distributions of the copula model
        :returns the individual uniqueness score of the single record
        """
        if self.fitted_model is None:
            raise Exception("Please fit the model first.")
        return self.correctmatch.individual_uniqueness(
            self.fitted_model, self.map_record_to_copula(row), self.size
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the individual privacy risk scores of the original records

        :param df: pd.DataFrame of the real records to be scored
        :returns pd.Series of the individual uniqueness scores for all records in the df.
        """
        transformed_df = self.map_records_to_copula(df)
        return pd.Series(
            self.predict_transformed(transformed_df),
            index=df.index,
            name="individual_uniqueness_score",
        )

    def map_records_to_copula(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert real dataset values of a dataframe to values that correspond
        to the marginal distributions of the fitted copula using the look up dictionary.

        :param df: pd.DataFrame of the real records to be scored
        :returns pd.DataFrame of the transformed records, their values correpond to their
        representations of the marginal distributions of the copula model.
        """
        if self.fitted_model is None:
            raise Exception("Please fit the model first.")
        transformed_df = pd.concat(
            [
                df[col].replace(self.look_up[col]).to_frame()
                for col in df.columns
            ],
            axis=1,
        )
        return transformed_df

    def score_func(self, transformed_row: List[int]) -> float:
        """Estimate individual uniquess for a tranformed record.

        :param transformed_row: List of integer values that correpond to the
        representation of the record given the marginal distributions of the copula model.
        :returns the individual uniqueness score of the single transformed
        record
        """
        if self.fitted_model is None:
            raise Exception("Please fit the model first.")
        return self.correctmatch.individual_uniqueness(
            self.fitted_model,
            transformed_row,
            self.size,
        )

    def predict_transformed(self, transformed_df: pd.DataFrame) -> np.ndarray:
        """Return the privacy risk scores of the transformed records

        :param transformed_df: pd. DataFrame of the transformed records to be scored
        :returns np.ndarray of the individual uniqueness scores for all transformed records
        in the transformed_df
        """
        return np.apply_along_axis(
            self.score_func, 1, transformed_df.astype("int")
        )

    def re_identify(
        self, individual_uniqueness: float, pop_size: int = 0
    ) -> float:
        """Calculate the likelihood of re-identifcation given the individual uniqueness

        :param individual_uniqueness: output of the get_individual_uniqueness
        :param pop_size: the size of the population
        :returns float result of the re-identifcation formula as described in the PCM paper
        """
        if pop_size == 0:
            if self.size == 0:
                raise Exception("Please fit model first")
            pop_size = self.size
        elif pop_size == 1:
            raise ValueError("Population size has to be higher than 1.")

        re_id = (1 / pop_size) * (
            (1 - individual_uniqueness ** (pop_size / (pop_size - 1)))
            / (1 - individual_uniqueness ** (1 / (pop_size - 1)))  # noqa: W503
        )
        return re_id
