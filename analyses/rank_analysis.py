import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import add
from helpers import get_colonist
from helpers import prepare_data
from scipy.stats import mannwhitneyu as mwu


def plot_ranks(ranked_data, crisis, countries, colors):
    plt.figure()
    plt.title(f"No. of years: {crisis.replace('_', ' ')}")
    plt.xlabel("Country")
    plt.ylabel("No. of years")
    countries, ranked_data = zip(*[(x, y) for y, x in
                                   sorted(zip(ranked_data, countries))])
    plt.bar(countries, ranked_data, color=colors)
    plt.savefig(f"rank_figs/ranked_data_{crisis}")


crises_df = pd.read_csv("../data/african_crises.csv")

# Columns that signify what rows/years had crises
crises_cols = ["systemic_crisis", "currency_crises", "inflation_crises",
               "banking_crisis"]

# Get the columns and start year we want
bool_crises_df = prepare_data(crises_df, crises_cols + ["cc3"], 1957)

# Our parameters for the amount of resamples
n, m = 10000, 10 ** 8

# Our chosen significance level and side of test
# greater signifies that y's distribution is greater than x's distribution
alpha = 0.05
alternative = "greater"

binom_dist = np.random.binomial(n, alpha, m)

# Gather all french and british former colony country codes
ccs = bool_crises_df["cc3"].unique()
ccs = [cc for cc in ccs if get_colonist(cc) != "PRT"]

fra_total = [0 for cc in ccs if get_colonist(cc) == "FRA"]
gbr_total = [0 for cc in ccs if get_colonist(cc) == "GBR"]

french_colonies = [cc for cc in ccs if get_colonist(cc) == "FRA"]
british_colonies = [cc for cc in ccs if get_colonist(cc) == "GBR"]

# For each country save the sum (amount of crisis years)
for crisis in crises_cols:
    fra_sample, gbr_sample = [], []
    countries, colors = [], []
    for cc in ccs:
        if get_colonist(cc) == "FRA":
            fra_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df["cc3"] == cc].sum())
            colors.append('b')
        if get_colonist(cc) == "GBR":
            gbr_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df["cc3"] == cc].sum())
            colors.append('r')

    plot_ranks(gbr_sample + fra_sample, crisis, ccs, colors)

    # Calculate the true left-sided p_value from the mann-whitney U test
    true_p_value = mwu(gbr_sample, fra_sample, alternative=alternative).pvalue

    # In case the p value is lower than alpha, we reject H1 and add
    # some noise to the bar heights using a multinomial distribution
    if true_p_value < alpha:
        # Gather the probabilities of our found sample
        comb_sample = fra_sample + gbr_sample
        comb_probs = np.array(comb_sample) / np.sum(comb_sample)
        resamples = np.random.multinomial(np.sum(comb_sample), comb_probs,
                                          size=n)
        resampled_p_values = [mwu(r[len(fra_sample):], r[:len(fra_sample)],
                                  alternative=alternative).pvalue
                              for r in resamples]
        # Count the amount of resampled p-values that are below alpha
        below_alpha_count = len([p for p in resampled_p_values if p <= alpha])
        plt.figure()
        plt.hist(binom_dist)
        plt.axvline(below_alpha_count, linestyle=":", label="count", color="g")
        plt.axvline(np.percentile(binom_dist, 2.5), linestyle="--",
                    label="95% CI", color="r")
        plt.axvline(np.percentile(binom_dist, 97.5), linestyle="--", color="r")
        plt.legend()
        plt.show()

    print(crisis, true_p_value)
