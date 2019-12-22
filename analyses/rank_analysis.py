import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import add
from helpers import get_colonist
from helpers import prepare_data
from scipy.stats import mannwhitneyu as mwu


def plot_ranks(dataframe, countries):

    # For each country plot the amount of years in crisis as a bar
    for crisis in dataframe:
        # Skip over the country code column
        if crisis == "cc3":
            continue

        ranked_data = []
        for cc in countries:
            ranked_data.append(dataframe[crisis]
                               .loc[dataframe["cc3"] == cc].sum())

            countries, ranked_data = zip(*[(x, y) for y, x in
                                   sorted(zip(ranked_data, countries))])

            colors = ["r" if get_colonist(cc) == "GBR" else "b" for cc in countries]

            plt.figure()
            plt.title(f"No. of years: {crisis.replace('_', ' ')}")
            plt.xlabel("Country")
            plt.ylabel("No. of years")
            plt.bar(countries, ranked_data, color=colors)
            plt.savefig(f"rank_figs/ranked_data_{crisis}")
            plt.show()


def prepare_data():

    crises_df = pd.read_csv("../data/african_crises.csv")

    # Columns that signify what rows/years had crises
    crises_cols = ["systemic_crisis", "currency_crises", "inflation_crises",
                   "banking_crisis"]

    # Replace values in banking_crisis with boolean values
    crises_df = crises_df.replace({"banking_crisis":
                                   {"crisis": 1, "no_crisis": 0}})
    crises_df = crises_df[crises_df["year"] > 1957]

    # Gather all boolean crises from after 1957
    return crises_df[crises_cols + ["cc3"]], crises_df["cc3"].unique()


# Gather our dataframe and the countries we have minus the portuguese colonies
crises_df, ccs = prepare_data()
ccs = [cc for cc in ccs if get_colonist(cc) != "PRT"]

# Plot the bar plots that vizualise the data the mwu-test is performed over
plot_ranks(crises_df, ccs)


# Perform the mann whitney u test for each tail
for alternative in ["less", "greater"]:
    for crisis in crises_df:
        # Skip over country code column
        if crisis == "cc3":
            continue

        fra_sample, gbr_sample = [], []
        # Gather each countries data
        for cc in ccs:
            if get_colonist(cc) == "FRA":
                fra_sample.append(crises_df[crisis]
                                  .loc[crises_df["cc3"] == cc].sum())
            elif get_colonist(cc) == "GBR":
                gbr_sample.append(crises_df[crisis]
                                  .loc[crises_df["cc3"] == cc].sum())

        # Print our results and hypotheses
        print(f"H0: {crisis} distribution is the same for former British " +
              "and French colonies.")
        print(f"H1: {crisis} distribution is {alternative} for former " +
              "British colonies compared to former French colonies.")
        print("Note: In our case a greater distribution would equal " +
              "a less stable economy")
        print()
        mwuresult = mwu(gbr_sample, fra_sample, alternative=alternative)
        print("U statistic:", mwuresult.statistic)
        print("P-value:    ", mwuresult.pvalue)
        print()
