from typing import Callable, Tuple

import numpy as np
import pandas as pd
import shap


class PrivacyRiskExplainer:
    """An explainer using the SHAP python library that explains the
    outputs of the PyCorrectMatch model. It requires the risk scoring
    function and the number of features/columns of the dataframe to be
    explained.
    The number of features is required to create a mask, which is the
    baseline vector that representes the most common record in the copula
    model representation. The mask is a vector of 1s of size 1-by-N the
    number of features.
    """

    def __init__(self, prediction: Callable, n_features: int):
        self.prediction = prediction
        self.mask = np.array([1] * n_features).reshape(1, n_features)
        self.unmasked_index = []
        self.explainer = shap.Explainer(self.prediction, self.mask)

    def explain(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, shap._explanation.Explanation]:
        """Calculate both local and global shapley values on a dataframe
        that has been transformed so that its actual values are replaced
        by the values that match the marginal representation of the copula
        model.

        :params data: transformed pd.DataFrame of records
        :returns:
        pd.DataFrame with local shaply values
        pd.Series with global shap values
        shap object that facilitates the plotting functions"""
        orig_index = data.index.tolist()
        masked_records = pd.Series(self.mask[0], index=data.columns)
        mask_index = data.loc[
            data.apply(lambda x: x.equals(masked_records), axis=1)
        ].index.tolist()
        self.unmasked_index = [i for i in orig_index if i not in mask_index]
        explanation = self.explainer(data.loc[self.unmasked_index])
        if len(mask_index) > 0:
            local_explanation_values = pd.concat(
                [
                    pd.DataFrame(
                        explanation.values,
                        index=self.unmasked_index,
                        columns=data.columns,
                    ),
                    pd.DataFrame(
                        np.zeros((len(mask_index), data.shape[1])),
                        index=mask_index,
                        columns=data.columns,
                    ),
                ],
                axis=0,
            )
        else:
            local_explanation_values = pd.DataFrame(
                explanation.values,
                index=self.unmasked_index,
                columns=data.columns,
            )
        global_explanation = local_explanation_values.mean()
        return (
            local_explanation_values.loc[orig_index],
            global_explanation,
            explanation,
        )

    def plot_local_explanation(
        self,
        explanation: shap._explanation.Explanation,
        orig_index: int,
        max_display: int = 10,
        show: bool = True,
    ):
        """Plot shapley values for an individual record.
        Note that is does not plot anything for the baseline
        records, i.e. the records that their values are all 1s
        once transformed to correspond to the copula representation."""
        if not self.unmasked_index:
            raise ("Call the explain function first")
        try:
            iloc = self.unmasked_index.index(orig_index)
            shap.plots.waterfall(explanation[iloc], max_display, show)
        except ValueError as v_error:
            print(v_error, "or record is baseline record.")

    def plot_global_explanation(
        self, explanation: shap._explanation.Explanation
    ):
        """Plot plot the mean absolute shap value for each feature column
        as a bar chart of one or many records"""
        shap.plots.bar(explanation)
