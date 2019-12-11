import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import add
from helpers import get_colonist
from scipy.stats import mannwhitneyu


crises_df = pd.read_csv('../data/african_crises.csv')

# Replace values in banking_crisis with boolean values
crises_df = crises_df.replace({'banking_crisis':
                               {'crisis': 1, 'no_crisis': 0}})

# Columns that signify what rows/years had crises
crises_cols = ['systemic_crisis', 'currency_crises', 'inflation_crises',
               'banking_crisis']


crises_df_after_1957 = crises_df.loc[crises_df['year'] > 1957]

# Gather all boolean crises from after 1957
bool_crises_df = crises_df_after_1957[['systemic_crisis', 'currency_crises',
                                       'inflation_crises', 'banking_crisis',
                                       'cc3']]

# Our parameters for the amount of resamples
n, m = 10000, 10 ** 8

# Our chosen significance level
alpha = 0.05

alternative = 'greater'

# binom_dist = np.random.binomial(1000, alpha, m)

ccs = crises_df['cc3'].unique()

fra_total = [0 for cc in ccs if get_colonist(cc) == 'FRA']
gbr_total = [0 for cc in ccs if get_colonist(cc) == 'GBR']

# For each country save the sum (amount of crisis years)
for crisis in crises_cols:
    fra_sample, gbr_sample = [], []
    for cc in ccs:
        if get_colonist(cc) == 'FRA':
            fra_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())
        if get_colonist(cc) == 'GBR':
            gbr_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())

    fra_total = [sum(x) for x in zip(fra_total, fra_sample)]
    gbr_total = [sum(x) for x in zip(gbr_total, gbr_sample)]

    # Calculate the true left-sided p_value from the mann-whitney U test
    true_p_value = mannwhitneyu(gbr_sample, fra_sample,
                                alternative=alternative).pvalue
    print(crisis, true_p_value)

# Calculate the true left-sided p_value from the mann-whitney U test of
# the sum of our samples
true_p_value = mannwhitneyu(gbr_total, fra_total,
                            alternative=alternative).pvalue
print('summed total', true_p_value)
