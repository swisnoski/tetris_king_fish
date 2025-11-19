"""
Utility functions
"""

from math import degrees, radians


def rad2deg(rad_list):
    """
    Take a list of angles in radians and convert them to degrees

    Args:
        rad_list: [list of radians float values]

    Returns:
        [list of degree values]
    """
    return [degrees(x) for x in rad_list]


def deg2rad(deg_list):
    """
    Take a list of angles in degrees adn convert them to radians

    Args:
        deg_list: [list of degree float/int values]

    Returns:
        [list of radian values]
    """
    return [radians(x) for x in deg_list]
