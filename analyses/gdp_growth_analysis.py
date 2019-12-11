import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast
# from sklearn import linear_model
import statsmodels.api as sm
# from scipy import stats

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

with open('../data/AfricanCountriesName.txt', 'r') as names:
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
    # print(growth_stats)
    growth_percol = [growth_France, growth_GB, growth_All]

    jellepelle = stats.ttest_ind(growth_France, growth_GB, axis=0, equal_var=True, nan_policy='propagate')
    print(jellepelle)

    return GDPgrowth_percountry, growth_percol, growth_stats


def regress_growth(GDPgrowth_percountry):
    """
    Regresses the growth rates of countries
    """
    corr_wld = []
    pval_wld = []
    corr_afr = []
    pval_afr = []

    for country in ['FRA', 'GBR', 'WLD']:
        corr = []
        pval = []
        for Afrcountry in col_France_GB:
            regress = stats.linregress(GDPgrowth_percountry[country], GDPgrowth_percountry[Afrcountry])
            r_value = regress[2]
            p_value = regress[3]
            corr.append(r_value)
            pval.append(p_value)

        corr_wld.append(corr)
        pval_wld.append(pval)


    corr_wld=np.array(corr_wld)
    pval_wld=np.array(pval_wld)

    for Afrcountry1 in col_France_GB:
        corr = []
        pval = []
        for Afrcountry2 in col_France_GB:
            regress = stats.linregress(GDPgrowth_percountry[Afrcountry1], GDPgrowth_percountry[Afrcountry2])
            r_value = regress[2]
            p_value = regress[3]
            corr.append(r_value)
            pval.append(p_value)

        corr_afr.append(corr)
        pval_afr.append(pval)

    corr_afr = np.array(corr_afr)
    pval_afr = np.array(pval_afr)

    return corr_wld, pval_wld, corr_afr, pval_afr


def regres_model(GDPgrowth_percountry):
    """
    Runs a mulivariate regression for GDP growth rates
    """
    growth_WLD = GDPgrowth_percountry['WLD'].tolist()
    del growth_WLD[0]
    del growth_WLD[0]
    growth_FRA = GDPgrowth_percountry['FRA'].tolist()
    del growth_FRA[0]
    del growth_FRA[0]
    growth_GBR = GDPgrowth_percountry['GBR'].tolist()
    del growth_GBR[0]
    del growth_GBR[0]

    df = pd.read_csv(worldtradedata)
    tradevol = df['value'].tolist()
    tradegrowth = [100*(tradevol[i+3]-tradevol[i+2])/tradevol[i+2] for i in range(len(tradevol)-3)]

    all_regres = {}
    growth_dat = {}
    for country in GDPgrowth_percountry:
        complete_data = True
        for gdpval in GDPgrowth_percountry[country]:
            if str(gdpval) == 'nan':
                complete_data = False

        if complete_data:
            if country in col_France_GB:
            # collect relevant data
                growth = GDPgrowth_percountry[country].tolist()
                del growth[0]
                del growth[0]
                growth_1 = GDPgrowth_percountry[country].tolist()
                del growth_1[0]
                del growth_1[-1]
                growth_2 = GDPgrowth_percountry[country].tolist()
                del growth_2[-1]
                del growth_2[-1]

                data = {'growth': growth, 'growth_1': growth_1, 'growth_2': growth_2,
                        'WLD': growth_WLD, 'FRA': growth_FRA, 'GBR': growth_GBR,
                        'Trade': tradegrowth}

                # run multiple linear regression
                df = pd.DataFrame(data, columns=['growth', 'growth_1', 'growth_2',
                                                    'WLD', 'FRA', 'GBR', 'Trade'])
                # df = pd.DataFrame(data, columns=['growth', 'growth_1', 'growth_2',
                                                    # 'WLD', 'FRA'])

                X = df[['growth_1', 'growth_2', 'WLD', 'FRA', 'GBR', 'Trade']]
                # X = df[['growth_1', 'growth_2', 'WLD', 'FRA']]
                Y = df['growth']
                X2 = sm.add_constant(X)
                regr = sm.OLS(Y, X2)
                regr = regr.fit()

                all_regres[country] = {'a': regr.params['const'], 'b1': regr.params['growth_1'],
                                        'b2': regr.params['growth_2'], 'b3': regr.params['WLD'],
                                        'b4': regr.params['FRA'], 'b5': regr.params['GBR'],
                                        'b6': regr.params['Trade']}

                print(AfricanCountriesName[AfricanCountriesCode.index(country)], country)
                print(regr.summary())

                growth_dat[country] = [growth, growth_1, growth_2]


    print(len(growth_WLD))
    used_dat = [growth_WLD, growth_FRA, growth_GBR, growth_dat, tradegrowth]

    return all_regres, used_dat


def vis_growth(GDPgrowth_percountry, growth_percol, growth_stats):
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

    time = np.arange(startyear, endyear+1, 1)

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
            return 0.2
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


def vis_regress_heat(corr_wld, pval_wld, corr_afr, pval_afr):
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

    plt.figure()
    plt.imshow(corr_afr)
    plt.yticks(range(len(countries_name)), countries_name)
    plt.xticks(range(len(countries_name)), countries_name, rotation='vertical')
    plt.title(f'GDP growth correlation in African countries for {startyear} - {endyear}')
    plt.savefig('gdp_growth_figs/gdp_regress_AFR.png')


def vis_regress_model(all_regres, GDPgrowth_percountry, used_dat):
    """
    Visualises time series with estimated model.
    """
    # get used data
    country = 'KEN'
    rval = all_regres[country]
    WLD = used_dat[0]
    FRA = used_dat[1]
    GBR = used_dat [2]
    tradegrowth = used_dat[4]
    growth_dat = used_dat[3][country]
    growth = growth_dat[0]
    growth_1 = growth_dat[1]
    growth_2 = growth_dat[2]

    # set timescales
    # time = np.arange(startyear, endyear + 1, 1)
    time = np.arange(startyear + 2, endyear + 1, 1)

    # get model parameters
    a = rval['a']
    b1 = rval['b1']
    b2 = rval['b2']
    b3 = rval['b3']
    b4 = rval['b4']
    b5 = rval['b5']
    b6 = rval['b6']

    # estimate values using parameters in data
    est_val = [a+b1*growth_1[i]+b2*growth_2[i]+b3*WLD[i]+b4*FRA[i]+b5*GBR[i]
                +b6*tradegrowth[i] for i in range(len(WLD))]

    country = AfricanCountriesName[AfricanCountriesCode.index(country)]

    plt.figure()
    plt.plot(time, growth, label='Real growth')
    plt.plot(time, est_val, label='Estimated growth')
    plt.legend()
    plt.title(f'Real and estimated values of GDP growth for {country}.')
    plt.xlabel('time')
    plt.ylabel('GDP growth')
    plt.show()


if __name__ == '__main__':

    GDPgrowth_percountry, growth_percol, growth_stats = GDPgrowth(AfricanCountriesCode)

    # vis_growth(GDPgrowth_percountry, growth_percol, growth_stats)

    # corr_wld, pval_wld, corr_afr, pval_afr = regress_growth(GDPgrowth_percountry)

    all_regres, used_dat = regres_model(GDPgrowth_percountry)

    # vis_regress_heat(corr_wld, pval_wld, corr_afr, pval_afr)

    vis_regress_model(all_regres, GDPgrowth_percountry, used_dat)
