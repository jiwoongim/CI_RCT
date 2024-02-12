import numpy as np

from ci.util_stats import get_confidence_interval, get_standard_error

from . import DF

ONLINE = 0
INCLASS = 1


class OnlineClassRoom:
    def __init__(self, dataframe: DF):
        self.data_df = dataframe
        self.outcomes = None

    def get_treatment_stats(self):
        onlineclass = np.sum(
            (self.data_df["format_ol"] == 1) * (self.data_df["format_blended"] == 0)
        )
        inclass = np.sum(
            (self.data_df["format_ol"] == 0) * (self.data_df["format_blended"] == 0)
        )
        blended = np.sum(self.data_df["format_blended"] == 1)

        print(f"Number of online class: {onlineclass}")
        print(f"Number of in-person class: {inclass}")
        print(f"Number of blended: {blended}")

    def get_potential_outcomes_by_treatment(self):
        # numpy select return an array drawn from elements in choicelist, depending on conditions.
        condition = [
            self.data_df["format_ol"].astype(bool),
            self.data_df["format_blended"].astype(bool),
        ]
        choice = ["online", "blended"]
        subset_by_treamtents = np.select(condition, choice, default="inclass")
        formatted = self.data_df.assign(class_format=subset_by_treamtents)
        return formatted.groupby(["class_format"]).mean()

    def get_outcomes_by_treatment(self):
        online_outcomes = self.data_df[self.data_df["format_ol"] == 1]["falsexam"]
        inclass_mask = (self.data_df["format_ol"] == 0) * (
            self.data_df["format_blended"] == 0
        )
        inclass_outcomes = self.data_df[inclass_mask]["falsexam"]
        self.outcomes = [online_outcomes, inclass_outcomes]

    def get_se_of_outcomes_by_treatment(self):
        if self.outcomes == None:
            self.get_outcomes_by_treatment()

        online_se = get_standard_error(self.outcomes[ONLINE])
        inclass_se = get_standard_error(self.outcomes[INCLASS])

        return online_se, inclass_se

    def get_ci_of_outcomes_by_treatment(self):
        if self.outcomes == None:
            self.get_outcomes_by_treatment()

        online_ci = get_confidence_interval(self.outcomes[ONLINE])
        inclass_ci = get_confidence_interval(self.outcomes[INCLASS])

        return online_ci, inclass_ci

    def get_ci_of_average_treatment_effect(self, c=1.96):
        if self.outcomes == None:
            self.get_outcomes_by_treatment()

        ate = self.outcomes[ONLINE].mean() - self.outcomes[INCLASS].mean()
        ses = self.get_se_of_outcomes_by_treatment()
        se_ate = np.sqrt(ses[ONLINE] ** 2 + ses[INCLASS] ** 2)
        return (ate - c * se_ate, ate + c * se_ate), ate
