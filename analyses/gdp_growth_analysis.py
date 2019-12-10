import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast

import helpers

"""
SDA Project of Daniel Perez Jensen, Jelle Mouissie and Joos Akkerman

This file collects nominal GDP growth data, fits distribution to growth rates
per former colonial motherland, runs regressions between growth rates and
plots results
"""

GDPgrowthdata = '../data/GDPgrowth_Worldbank.csv'
startyear = 1961
endyear = 2014

# collect country and category data from text files and save as global
with open('../data/AfricanCountriesCode.txt', 'r') as codes:
    codes = codes.read()
    AfricanCountriesCode = ast.literal_eval(codes)

with open('../data/AfricanCountriesCode.txt', 'r') as names:
    names = names.read()
    AfricanCountriesName = ast.literal_eval(names)

with open('../data/col_names.txt', 'r') as all_col:
    all_col = all_col.read()
    all_col = ast.literal_eval(all_col)

col_France = all_col['France']
col_GB = all_col['GB']

col_France_GB = col_France + col_GB


def GDPgrowth(AfricanCountriesCode):
    """
    Extracts growth rate per country
    """
    df = pd.read_csv(GDPgrowthdata)
    years = np.arange(startyear, endyear+1).tolist()
    GDPgrowth_percountry = {}
    growth_France = []
    growth_GB = []
    growth_All = []

    for country in ['FRA', 'GBR', 'WLD']:
        df2 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
        df2 = df2.values[0]
        GDPgrowth_percountry[country] = df2

    for country in AfricanCountriesCode:
        if country in col_France or country in col_GB:
            df3 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
            df3 = df3.values[0]

            GDPgrowth_percountry[country] = df3

    # categorize growth numbers by colonial overlord:
    nFrance = 0
    nGB = 0
    for country in GDPgrowth_percountry:
        # check if data is complete
        complete_data = True
        for gdpval in GDPgrowth_percountry[country]:
            if str(gdpval) == 'nan':
                complete_data = False

        if complete_data:
            growth_All += GDPgrowth_percountry[country].tolist()
            if country in col_France:
                nFrance += 1
                growth_France += GDPgrowth_percountry[country].tolist()
            elif country in col_GB:
                nGB += 1
                growth_GB += GDPgrowth_percountry[country].tolist()

    # calculate stats to fit normal distribution:
    avg_France = np.mean(growth_France)
    stddev_France = np.std(growth_France)
    avg_GB = np.mean(growth_GB)
    stddev_GB = np.std(growth_GB)
    avg_All = np.mean(growth_All)
    stddev_All = np.std(growth_All)

    statFrance = {'avg': avg_France, 'stddev': stddev_France, 'n_countries': nFrance, 'n_sample': len(growth_France)}
    statGB = {'avg': avg_GB, 'stddev': stddev_GB, 'n_countries': nGB, 'n_sample': len(growth_GB)}
    statAll = {'avg': avg_All, 'stddev': stddev_All, 'n_countries': nGB + nFrance, 'n_sample': len(growth_All)}

    growth_stats = [statFrance, statGB, statAll]
    print(growth_stats)
    growth_percol = [growth_France, growth_GB, growth_All]

    jellepelle = stats.ttest_ind(growth_France, growth_GB, axis=0, equal_var=True, nan_policy='propagate')
    print(jellepelle)

    return GDPgrowth_percountry, growth_percol, growth_stats


def regress_growth(GDPgrowth_percountry):
    """
    Regresses the growth rates of countries
    """
    print(stats.linregress(GDPgrowth_percountry['GBR'], GDPgrowth_percountry['FRA']))
    print(stats.linregress(GDPgrowth_percountry['GBR'], GDPgrowth_percountry['WLD']))
    print(stats.linregress(GDPgrowth_percountry['GBR'], GDPgrowth_percountry['NGA']))
    print(stats.linregress(GDPgrowth_percountry['WLD'], GDPgrowth_percountry['NGA']))

    corr_wld = []
    all_pval = []


    for country in ['FRA', 'GBR', 'WLD']:
        corr = []
        pval = []
        for Afrcountry in col_France_GB:
            regress = stats.linregress(GDPgrowth_percountry[country], GDPgrowth_percountry[Afrcountry])
            r_value = regress[2]
            p_value = regress[3]
            corr.append(r_value)
            pval.append(p_value)
            # print(f'{Afrcountry} and {country}:')
            # print(r_value, p_value)
        corr_wld.append(corr)
        all_pval.append(pval)


    corr_wld=np.array(corr_wld)
    all_pval=np.array(all_pval)
    print(corr_wld)
    print(all_pval)

    # for Afrcountry1 in col_France_GB:
    #     for Afrcountry2 in col_France_GB:
    #         regress = stats.linregress(GDPgrowth_percountry[Afrcountry1], GDPgrowth_percountry[Afrcountry2])
    #         r_value = regress[2]
    #         p_value = regress[3]
    #         print(f'{Afrcountry1} and {Afrcountry2}:')
    #         print(r_value, p_value)

    return corr_wld, all_pval


def vis_growth(growth_percol, growth_stats):
    """
    Visualizes distribution of growth numbers
    """
    statFrance = growth_stats[0]
    statGB = growth_stats[1]
    statAll = growth_stats[2]

    plt.figure()
    x = np.linspace(statAll['avg'] - 4*statAll['stddev'], statAll['avg'] + 4*statAll['stddev'], 100)
    plt.plot(x, stats.norm.pdf(x, statFrance['avg'], statFrance['stddev']), label='Fitted normal distribution France', color='blue')
    plt.plot(x, stats.norm.pdf(x, statGB['avg'], statGB['stddev']), label='Fitted normal distribution Great_Britain', color='red')
    plt.hist(growth_percol[0], 100, density=True, color='blue', alpha=0.5, label='Growth rates French ex-colonies')
    plt.hist(growth_percol[1], 100, density=True, color='red', alpha=0.5, label='Growth rates British ex-colonies')
    plt.axvline(x=statGB['avg'], color='red')
    plt.axvline(x=statFrance['avg'], color='blue')
    plt.title(f'Distribution of anual growth rates, {startyear} - {endyear}.')
    plt.ylabel('Frequency')
    plt.xlabel('Annual growth rate of GDP')
    plt.legend(loc='upper left')
    plt.savefig('gdp_growth_figs/gdp_growth.png')


def vis_regress(corr_wld, all_pval):
    """
    Visualized regression values in heat map
    """

    countries_name = [AfricanCountriesName[AfricanCountriesCode.index(el)] for el in col_France_GB]

    wld = ['France', 'Great Britain', 'World']
    plt.figure()
    plt.imshow(corr_wld)
    plt.yticks([0,1,2], wld)
    plt.xticks(range(len(countries_name)), countries_name, rotation='vertical')
    plt.title(f'GDP growth correlation for {startyear} - {endyear}')
    plt.savefig('gdp_growth_figs/gdp_regress_WLD.png')


if __name__ == '__main__':

    GDPgrowth_percountry, growth_percol, growth_stats = GDPgrowth(AfricanCountriesCode)

    vis_growth(growth_percol, growth_stats)

    corr_wld, all_pval = regress_growth(GDPgrowth_percountry)

    vis_regress(corr_wld, all_pval)
