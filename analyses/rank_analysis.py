import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
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

n = 10000

fra_sample, gbr_sample = [], []
ccs = crises_df['cc3'].unique()

# For each country save the sum (amount of crisis years)
for crisis in crises_cols:
    for cc in ccs:
        if get_colonist(cc) == 'FRA':
            fra_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())
        if get_colonist(cc) == 'GBR':
            gbr_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())

    # Calculate the true left-sided p_value from the mann-whitney U test
    true_p_value = mannwhitneyu(fra_sample, gbr_sample,
                                alternative='greater').pvalue

    # Resample using a multinomial distribution
    full_sample = fra_sample + gbr_sample
    full_sample_probs = np.array(full_sample) / np.sum(full_sample)
    resamples = np.random.multinomial(np.sum(full_sample),
                                      full_sample_probs, size=n)

    # Run our pairwise test on each of the n new samples
    resampled_mannwhitneyu = [mannwhitneyu(r[:len(fra_sample)], r[len(fra_sample):],
                              alternative='greater').pvalue for r in resamples]

    # Plotting stuff
    plt.figure()
    plt.title(f'left sided test p-values of {crisis.replace("_", " ")}')
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.hist(resampled_mannwhitneyu, bins=int(np.sqrt(n)))

    # Plot our true p_value
    plt.axvline(true_p_value, c='k', linestyle='--',
                label='true p-value')
    plt.axvline(np.mean(resampled_mannwhitneyu), c='m', linestyle=':',
                label='mean')

    plt.legend()
    plt.savefig(f'rank_figs/{crisis.replace("_", " ")}-pval-dist')
