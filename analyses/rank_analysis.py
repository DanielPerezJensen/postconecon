import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random


def pairwise_test(x, y, n):
    """Draw a sample from x and y, n times and compare their values.
    Returns the fraction of times that the value drawn from x is larger
    than the value drawn from y."""
    count = 0
    for _ in range(n):
        x_random = random.choice(x)
        y_random = random.choice(y)
        if x_random > y_random:
            count += 1
    return count / n


def get_colonist(cc3):
    """Returns the 3 letter country code of the colonizer of cc3"""
    english_colonies = ['EGY', 'KEN', 'NGA', 'ZAF', 'ZMB', 'ZWE', 'MUS']
    french_colonies = ['DZA', 'CIV', 'CAF', 'MAR', 'TUN']
    portuguese_colonies = ['AGO']

    if cc3 in portuguese_colonies:
        return 'PRT'
    elif cc3 in english_colonies:
        return 'GBR'
    elif cc3 in french_colonies:
        return 'FRA'


crises_df = pd.read_csv('../data/african_crises.csv')

# Replace values in banking_crisis with boolean values
crises_df = crises_df.replace({'banking_crisis':
                               {'crisis': 1, 'no_crisis': 0}})

# Columns that signify what rows/years had crises
crises_cols = ['systemic_crisis', 'domestic_debt_in_default',
               'currency_crises', 'inflation_crises',
               'banking_crisis']


crises_df_after_1957 = crises_df.loc[crises_df['year'] > 1957]

# Gather all boolean crises from after 1957
bool_crises_df = crises_df_after_1957[['systemic_crisis',
                                       'domestic_debt_in_default',
                                       'currency_crises', 'inflation_crises',
                                       'banking_crisis', 'cc3']]

# Repeat n pairwise tests m times
n, m = 10000, 1000

# If the French ranks aren't statistically higher ranked than the British
# we would expect the mean of our test to fall within the 95%-CI of
# a binomial distribution with p=0.5.
binomial_samples = [np.random.binomial(n, 0.5) / n for _ in range(m)]

FRA_sample, GBR_sample = [], []
ccs = crises_df['cc3'].unique()

# For each country save the sum (amount of crisis years)
for crisis in crises_cols:
    for cc in ccs:
        if get_colonist(cc) == 'FRA':
            FRA_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())
        if get_colonist(cc) == 'GBR':
            GBR_sample.append(bool_crises_df[crisis]
                              .loc[bool_crises_df['cc3'] == cc].sum())

    # Calculate the mean of our pairwise_test
    true_pairwise_mean = np.mean([pairwise_test(FRA_sample, GBR_sample, n)
                                  for _ in range(m)])

    # Resample our data using a multinomial distribution
    gbr_p_values = np.array(GBR_sample) / len(bool_crises_df)
    fra_p_values = np.array(FRA_sample) / len(bool_crises_df)

    # TODO: Not sure if it should be length of sum
    gbr_resamples = np.random.multinomial(len(bool_crises_df), gbr_p_values,
                                          size=50)
    fra_resamples = np.random.multinomial(len(bool_crises_df), fra_p_values,
                                          size=50)

    # Zip our samples together so we can use it with our pairwise test function
    zipped_resamples = zip(fra_resamples, gbr_resamples)

    # Run our pairwise test on each of the 50 new samples
    resampled_pairwise_tests = [pairwise_test(f, g, n) for
                                f, g in zipped_resamples]

    # Plotting stuff
    plt.title(crisis)
    plt.xlabel('Percentage FRA > GBR')
    plt.ylabel('Count')
    plt.hist(binomial_samples)

    # Plot confidence interval and our calculated mean
    plt.axvline(true_pairwise_mean, c='k', linestyle='--',
                label='true pairwise mean')
    plt.axvline(np.percentile(binomial_samples, 1.25), color='g',
                label='CI', linestyle=':')
    plt.axvline(np.percentile(binomial_samples, 97.5), color='g',
                linestyle=':')

    # Plot each of our resamples in magenta and low opaqueness
    for resampled_test in resampled_pairwise_tests:
        plt.axvline(resampled_test, c='m', linestyle='-', alpha=0.25)
    plt.legend()
    plt.show()
