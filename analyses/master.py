"""This file contains a number of functions that
are used across multiple analyses"""


def get_colonist(cc3):
    """Returns the 3 letter country code of the colonizer of cc3"""
    colonists_dict = {'GBR': ['EGY', 'KEN', 'NGA', 'ZAF', 'ZMB', 'ZWE', 'MUS'],
                      'FRA': ['DZA', 'CIV', 'CAF', 'MAR', 'TUN'],
                      'PRT': ['AGO']}

    for colonist, colonized_list in colonists_dict.items():
        if cc3 in colonized_list:
            return colonist

    return False
