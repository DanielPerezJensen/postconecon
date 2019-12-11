import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast

import helpers

"""
SDA Project of Daniel Perez Jensen, Jelle Mouissie and Joos Akkerman

This file collects nominal GDP data, computes indeces, plots results and
performs statistical tests on indeces.
"""

GDPdata = 'GDP_Worldbank.csv'
startyear = 1960
endyear = 2014

# collect country and category data from text files and save as global
with open('AfricanCountriesCode.txt', 'r') as codes:
    codes = codes.read()
    AfricanCountriesCode = ast.literal_eval(codes)

with open('AfricanCountriesCode.txt', 'r') as names:
    names = names.read()
    AfricanCountriesName = ast.literal_eval(names)

with open('col_names.txt', 'r') as all_col:
    all_col = all_col.read()
    all_col = ast.literal_eval(all_col)

col_France = all_col['France']
col_GB = all_col['GB']

col_France_GB = col_France + col_GB


def indexGDP(AfricanCountriesCode):
    """
    Calculates indexes for GDP size per country
    """
    df = pd.read_csv(GDPdata)
    years = np.arange(startyear, endyear+1).tolist()
    GDPdata_index = {}
    GDPdata_percol = {'France': [0 for el in range(len(years))], 'GB': [0 for el in range(len(years))]}


    def indexnumber(list):
        return [100*list[i]/list[0] for i in range(len(list))]


    for country in ['FRA', 'GBR', 'WLD']:
        df2 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
        df2 = df2.values[0]
        indexes = indexnumber(df2)
        GDPdata_index[country] = indexes

    # add data for each country that has been colony of France or GB
    for country in AfricanCountriesCode:
        if country in col_France or country in col_GB:
            df2 = df.loc[df['Country Code'] == country, str(years[0]):str(years[-1])]
            df2 = df2.values[0]

            # check if data is complete
            complete_data = True
            for gdpval in df2:
                if str(gdpval) == 'nan':
                    complete_data = False

            if complete_data:
                if country in col_France:
                    for i in range(len(years)):
                        GDPdata_percol['France'][i] += df2[i]
                elif country in col_GB:
                    for i in range(len(years)):
                        GDPdata_percol['GB'][i] += df2[i]

            indexes = indexnumber(df2)
            GDPdata_index[country] = indexes

    GDPdata_index_percol = [indexnumber(GDPdata_percol['France']), indexnumber(GDPdata_percol['GB'])]

    return GDPdata_index, GDPdata_index_percol, years


def vis_index(GDPdata_index, GDPdata_index_percol, years):
    """
    Visualizes indexes of GDP per colonial overlord
    """

    def color(country):
        if country in col_France:
            return 'blue'
        elif country in col_GB:
            return 'red'
        elif country == 'WLD':
            return 'green'
        elif country == 'FRA':
            return 'black'
        elif country == 'GBR':
            return 'purple'

    def alpha(country):
        if country in col_France or country in col_GB:
            print(country)
            return 0.2
        else:
            return 1.0

    for country in GDPdata_index:
        if country=='WLD' or country=='GBR' or country == 'FRA':
            plt.plot(years, GDPdata_index[country], color=color(country), alpha=1, label=f'{country}')
        else:
            plt.plot(years, GDPdata_index[country], color=color(country), alpha=0.2)

    plt.plot(years, GDPdata_index_percol[0], label='French average', color='blue')
    plt.plot(years, GDPdata_index_percol[1], label='British average', color='red')

    plt.yscale('log')
    plt.ylabel('GDP index')
    plt.xlabel('Time')
    plt.title('Indexes for GDP of african country, categorized by former colonial overlord.')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    GDPdata_index, GDPdata_index_percol, years = indexGDP(AfricanCountriesCode)

    vis_index(GDPdata_index, GDPdata_index_percol, years)
