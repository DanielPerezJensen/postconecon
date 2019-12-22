import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast
import statsmodels.api as sm

import helpers

"""
SDA Project of Daniel Perez Jensen, Jelle Mouissie and Joos Akkerman

This file collects nominal GDP growth data, fits distribution to growth rates
per former colonial motherland, runs regressions between growth rates and
plots results
"""

GDPgrowthdata = '../data/GDPgrowth_Worldbank.csv'
worldtradedata = '../data/worldtrade_data.csv'
startyear = 1961
endyear = 2014

# collect country and category data from text files and save as global
with open('../data/AfricanCountriesCode.txt', 'r') as codes:
    codes = codes.read()
    AfricanCountriesCode = ast.literal_eval(codes)

with open('../data/AfricanCountriesName.txt', 'r',
            encoding='utf-8', errors='ignore') as names:
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

    # extract growth data from csv
    for country in ['FRA', 'GBR', 'WLD']:
        df2 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
        df2 = df2.values[0]
        GDPgrowth_percountry[country] = df2

    for country in AfricanCountriesCode:
        if country in col_France_GB:
            df3 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
            df3 = df3.values[0]
            GDPgrowth_percountry[country] = df3

    # categorize growth numbers by colonizer:
    nFrance = 0
    nGB = 0
    complete_countries = []
    for country in GDPgrowth_percountry:

        # check if data is complete
        complete_data = True
        for gdpval in GDPgrowth_percountry[country]:
            if str(gdpval) == 'nan':
                complete_data = False

        if complete_data:
            if country in col_France:
                nFrance += 1
                complete_countries.append(country)
                growth_France += GDPgrowth_percountry[country].tolist()
                growth_All += GDPgrowth_percountry[country].tolist()
            elif country in col_GB:
                nGB += 1
                complete_countries.append(country)
                growth_GB += GDPgrowth_percountry[country].tolist()
                growth_All += GDPgrowth_percountry[country].tolist()

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
    growth_percol = [growth_France, growth_GB, growth_All]

    return GDPgrowth_percountry, growth_percol, growth_stats, complete_countries


def ttest(data1, data2, name1, name2):
    """
    Performs a t-test for the distribution of two samples
    """
    t_test = stats.ttest_ind(data1, data2, axis=0, equal_var=True, nan_policy='propagate')
    print(f'\n\nt-test for difference between {name1} and {name2}.')
    print(f'Results t-test:\n {t_test}\n\n')


def regress_growth(GDPgrowth_percountry, reg1, reg2):
    """
    Regresses the growth rates of countries
    """
    all_corr = []
    all_pval = []

    for country1 in reg1:
        corr = []
        pval = []
        for country2 in reg2:
            regress = stats.linregress(GDPgrowth_percountry[country1], GDPgrowth_percountry[country2])
            r_value = regress[2]
            p_value = regress[3]
            corr.append(r_value)
            pval.append(p_value)

        all_corr.append(corr)
        all_pval.append(pval)

    return np.array(all_corr), np.array(all_pval)


def regress_model(GDPgrowth_percountry):
    """
    Runs a mulivariate regression for GDP growth rates
    """
    growth_WLD = GDPgrowth_percountry['WLD'].tolist()
    del growth_WLD[0:2]
    growth_FRA = GDPgrowth_percountry['FRA'].tolist()
    del growth_FRA[0:2]
    growth_GBR = GDPgrowth_percountry['GBR'].tolist()
    del growth_GBR[0:2]

    all_regress = {}
    growth_dat = {}
    for country in GDPgrowth_percountry:
        complete_data = True
        for gdpval in GDPgrowth_percountry[country]:
            if str(gdpval) == 'nan':
                complete_data = False

        if complete_data:
            if country in col_France_GB:
                growth = GDPgrowth_percountry[country].tolist()
                del growth[0:2]
                growth_1 = GDPgrowth_percountry[country].tolist()
                del growth_1[0]
                del growth_1[-1]
                growth_2 = GDPgrowth_percountry[country].tolist()
                del growth_2[-1]
                del growth_2[-1]

                data = {'growth': growth, 'growth_1': growth_1, 'growth_2': growth_2,
                        'WLD': growth_WLD, 'FRA': growth_FRA, 'GBR': growth_GBR}

                # run multiple linear regression
                df = pd.DataFrame(data, columns=['growth', 'growth_1', 'growth_2',
                                                    'WLD', 'FRA', 'GBR'])

                # used for imperfect multicollinearity tests
                X = df[['growth_1', 'growth_2', 'WLD', 'FRA', 'GBR']]
                # X = df[['growth_1', 'growth_2', 'FRA', 'GBR']]
                # X = df[['growth_1', 'growth_2', 'WLD']]

                Y = df['growth']
                X2 = sm.add_constant(X)
                regr = sm.OLS(Y, X2)
                regr = regr.fit()

                all_regress[country] = {'a': regr.params['const'], 'pa': regr.pvalues['const'],
                                        'b1': regr.params['growth_1'], 'pb1': regr.pvalues['growth_1'],
                                        'b2': regr.params['growth_2'], 'pb2': regr.pvalues['growth_2'],
                                        'b3': regr.params['WLD'], 'pb3': regr.pvalues['WLD'],
                                        'b4': regr.params['FRA'], 'pb4': regr.pvalues['FRA'],
                                        'b5': regr.params['GBR'], 'pb5': regr.pvalues['GBR'],
                                        'R': regr.rsquared, 'adjR': regr.rsquared_adj}

                print(f'\nRegression Summary for {AfricanCountriesName[AfricanCountriesCode.index(country)]}, {country}:')
                print(regr.summary())

                growth_dat[country] = [growth, growth_1, growth_2]


    used_dat = [growth_WLD, growth_FRA, growth_GBR, growth_dat]

    return all_regress, used_dat


def countsig(all_regress):
    """
    Count all significant measurements and R squared values for regression models
    """
    sig_WLD_FRA = 0
    sig_FRA_FRA = 0
    sig_GBR_FRA = 0
    sig_WLD_GBR = 0
    sig_FRA_GBR = 0
    sig_GBR_GBR = 0
    R_FRA = 0
    adjR_FRA = 0
    R_GBR = 0
    adjR_GBR = 0

    for country in all_regress:
        if country in col_France:
            R_FRA += all_regress[country]['R']
            adjR_FRA += all_regress[country]['adjR']
            if all_regress[country]['pb3'] <= 0.05:
                sig_WLD_FRA += 1
            elif all_regress[country]['pb4'] <= 0.05:
                sig_FRA_FRA += 1
            elif all_regress[country]['pb5'] <= 0.05:
                sig_GBR_FRA += 1
        elif country in col_GB:
            R_GBR += all_regress[country]['R']
            adjR_GBR += all_regress[country]['adjR']
            if all_regress[country]['pb3'] <= 0.05:
                sig_WLD_GBR += 1
            elif all_regress[country]['pb4'] <= 0.05:
                sig_FRA_GBR += 1
            elif all_regress[country]['pb5'] <= 0.05:
                sig_GBR_GBR += 1

    R_FRA = R_FRA/13
    R_GBR = R_GBR/13
    adjR_FRA = adjR_FRA/13
    adjR_GBR = adjR_GBR/13

    print(f'R^2 for former French colonies is {R_FRA}.')
    print(f'Adjusted R^2 for former French colonies is {adjR_FRA}.')
    print(f'R^2 for former British colonies is {R_GBR}.')
    print(f'Adjusted R^2 for former British colonies is {adjR_GBR}.')
    print(f'sig FRA for FRA {sig_FRA_FRA}')
    print(f'sig FRA for WLD {sig_WLD_FRA}')
    print(f'sig FRA for GBR {sig_GBR_FRA}')
    print(f'sig GBR for FRA {sig_FRA_GBR}')
    print(f'sig GBR for WLD {sig_WLD_GBR}')
    print(f'sig GBR for GBR {sig_GBR_GBR}')


def vis_growth_hist(GDPgrowth_percountry, growth_percol, growth_stats):
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
    plt.savefig('gdp_growth_figs/gdp_growth_hist.png')


def vis_growth_line(GDPgrowth_percountry):
    """
    Visualises growth rates as time series
    """
    time = np.arange(startyear, endyear+1, 1)

    # set color based on country name
    def color(country):
        if country in col_France:
            return 'blue'
        elif country in col_GB:
            return 'red'
        elif country == 'WLD':
            return 'black'
        elif country == 'FRA':
            return 'blue'
        elif country == 'GBR':
            return 'red'

    def alpha(country):
        if country in col_France or country in col_GB:
            return 0.15
        else:
            return 1.0

    def label(country):
        if country =='WLD':
            return 'World'
        elif country == 'FRA':
            return 'France'
        elif country == 'GBR':
            return 'Great Britain'

    plt.figure()
    for country in GDPgrowth_percountry:
        plt.plot(time, GDPgrowth_percountry[country], color=color(country), alpha=alpha(country), label=label(country))
    plt.title('GDP growth rates over time')
    plt.xlabel('time')
    plt.ylabel('GDP growth rate')
    plt.legend(loc='lower left')
    plt.savefig('gdp_growth_figs/gdp_growth_time.png')


def vis_regress_heat(corr, pval, vert, name):
    """
    Visualized regression values in heat map
    """
    countries_name = [AfricanCountriesName[AfricanCountriesCode.index(el)] for el in col_France_GB]
    if name == 'AFR':
        vert = countries_name

    plt.figure()
    plt.imshow(corr)
    plt.yticks(range(len(vert)), vert)
    plt.xticks(range(len(countries_name)), countries_name, rotation='vertical')
    plt.title(f'GDP growth correlation in African countries for {startyear} - {endyear}')
    plt.savefig(f'gdp_growth_figs/gdp_regress_{name}.png')


def vis_regress_model(all_regress, GDPgrowth_percountry, used_dat, complete_countries):
    """
    Visualises time series with estimated model.
    """
    # set timescales
    time = np.arange(startyear + 2, endyear + 1, 1)

    for country in complete_countries:
        rval = all_regress[country]
        WLD = used_dat[0]
        FRA = used_dat[1]
        GBR = used_dat [2]
        growth_dat = used_dat[3][country]
        growth = growth_dat[0]
        growth_1 = growth_dat[1]
        growth_2 = growth_dat[2]

        # get model parameters
        a = rval['a']
        b1 = rval['b1']
        b2 = rval['b2']
        b3 = rval['b3']
        b4 = rval['b4']
        b5 = rval['b5']
        adjR = round(rval['adjR'], 2)

        # estimate values using parameters in data
        est_val = [a+b1*growth_1[i]+b2*growth_2[i]+b3*WLD[i]+b4*FRA[i]+b5*GBR[i]
                    for i in range(len(WLD))]

        # get country name
        country_name = AfricanCountriesName[AfricanCountriesCode.index(country)]

        # plot graphs and save
        plt.figure()
        plt.plot(time, growth, label='Real growth')
        plt.plot(time, est_val, label='Estimated growth')
        plt.legend()
        plt.title(f'Real and estimated values of GDP growth for {country_name} (adj R^2={adjR}).')
        plt.xlabel('time')
        plt.ylabel('GDP growth')
        plt.savefig(f'gdp_growth_figs/gdp_growth_model_{country}.png')


if __name__ == '__main__':

    GDPgrowth_percountry, growth_percol, growth_stats, complete_countries = GDPgrowth(AfricanCountriesCode)

    ttest(growth_percol[0], growth_percol[1], 'French ex-colonies', 'British ex-colonies')

    vis_growth_hist(GDPgrowth_percountry, growth_percol, growth_stats)
    vis_growth_line(GDPgrowth_percountry)

    corr_wld, pval_wld = regress_growth(GDPgrowth_percountry, ['FRA', 'GBR', 'WLD'], col_France_GB)
    corr_afr, pval_afr = regress_growth(GDPgrowth_percountry, col_France_GB, col_France_GB)

    vis_regress_heat(corr_wld, pval_wld, ['France', 'Great Britain', 'World'], 'WLD')
    vis_regress_heat(corr_afr, pval_afr, [], 'AFR')

    all_regress, used_dat = regress_model(GDPgrowth_percountry)

    countsig(all_regress)

    vis_regress_model(all_regress, GDPgrowth_percountry, used_dat, complete_countries)
