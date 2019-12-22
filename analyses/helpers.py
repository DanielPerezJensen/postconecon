import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast

GDPdataMeta = '../data/Metadata_worldbank.csv'

"""
SDA Project of Daniel Perez Jensen, Jelle Mouissie and Joos Akkerman

This file contains helper functions.
"""


def regions():
    """
    Gets meta data on region of countries
    """
    AfricanCountriesCode = []
    AfricanCountriesName = []
    df_meta = pd.read_csv(GDPdataMeta)

    rows = df_meta.iterrows()
    for index, row in df_meta.iterrows():
        if row['Region'] == 'Sub-Saharan Africa':
            AfricanCountriesCode.append(row['Country Code'])
            AfricanCountriesName.append(row['TableName'])

    # North Africa and Middle East are grouped together, so North African
    # countries have to be added manually (no data for Libya):
    NorthAfricaCodes = ['MAR', 'DZA', 'TUN', 'EGY']
    NorthAfricaNames = ['Morocco', 'Algeria', 'Tunisia', 'Egypt']

    AfricanCountriesCode += NorthAfricaCodes
    AfricanCountriesName += NorthAfricaNames

    filecode = open('AfricanCountriesCode.txt', 'w')
    filecode.write(str(AfricanCountriesCode))
    filecode.close()

    filename = open('AfricanCountriesName.txt', 'w')
    filename.write(str(AfricanCountriesName))
    filename.close()


def get_colonist(cc3):
    """
    Returns the 3 letter country code of the colonizer of cc3
    """
    colonists_dict = {'GBR': ['EGY', 'KEN', 'NGA', 'ZAF', 'ZMB', 'ZWE', 'MUS'],
                      'FRA': ['DZA', 'CIV', 'CAF', 'MAR', 'TUN'],
                      'PRT': ['AGO']}

    for colonist, colonized_list in colonists_dict.items():
        if cc3 in colonized_list:
            return colonist

    return False


def prepare_data(dataframe, columns, start_year):

    # Replace values in banking_crisis with boolean values
    dataframe = dataframe.replace({"banking_crisis":
                                   {"crisis": 1, "no_crisis": 0}})
    dataframe = dataframe[dataframe["year"] > start_year]

    # Gather all boolean crises from after 1957
    return dataframe[columns]
