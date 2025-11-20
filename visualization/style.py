from matplotlib.colors import LinearSegmentedColormap, to_rgb
from colorsys import rgb_to_hsv


"""
Module for visualization styles, including color schemes and markers.
"""

def get_color_method(name, color_scheme="default"):
    """
    Returns the RGB tuple for a given color name.

    Parameters:
    -----------
    name : str
        Name of the color.
    color_scheme : str
        Color scheme to use ("default" or "mdt-focus").
    Returns:
    --------
    tuple
        RGB values as a tuple.
    """
    if color_scheme == "default":
        colors = {
            "Random Convex MDT": "#2851CC",
            "Random MDT": "#4469DA",
            "Direct MDT": "#0B93C1",
            "Beam-Search MDT": "#07A0C3",
            "Alternating Diffusion": "#B9E28C",
            "Powered Alternating Diffusion": "#ACD86E",
            "Integrated Diffusion Maps": "#3BB273",
            "Multi-view Diffusion Maps": "#F7B5D4",
            "Cross Diffusion Maps": "#E67E22",
            "Composite Diffusion Maps": "#F49F0A",
        }
    elif color_scheme == "mdt-focus":
        colors = {
            "Random Convex MDT": "#2851CC",
            "Random MDT": "#4469DA",
            "Direct MDT": "#0B93C1",
            "Beam-Search MDT": "#07A0C3",
            # "Alternating Diffusion": "#B9E28C",
            # "Powered Alternating Diffusion": "#ACD86E",
            # "Integrated Diffusion Maps": "#3BB273",
            # "Multi-view Diffusion Maps": "#F7B5D4",
            # "Cross Diffusion Maps": "#E67E22",
            # "Composite Diffusion Maps": "#F49F0A",
        }
    elif color_scheme == "other-focus":
      colors = {
            # "Random Convex MDT": "#2851CC",
            # "Random MDT": "#4469DA",
            # "Direct MDT": "#0B93C1",
            # "Beam-Search MDT": "#07A0C3",
            "Alternating Diffusion": "#B9E28C",
            "Powered Alternating Diffusion": "#ACD86E",
            "Integrated Diffusion Maps": "#3BB273",
            "Multi-view Diffusion Maps": "#F7B5D4",
            "Cross Diffusion Maps": "#E67E22",
            "Composite Diffusion Maps": "#F49F0A",
        }
    return colors.get(name, "#7A7A7A")

def get_col_list():
    """
    Returns a list of predefined colors.
    """
    colors = [
        "#2851CC",
        "#F96C39",
        "#9A44C5",
        "#FFBF46",
        "#27A54F",
        "#ff459c",
        "#C9035F",
        "#1FD2FF",
        "#0086fb",
        "#D4D2D5",
        "#696A62",
    ]
    return colors


def get_marker_list():
    marks = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h', '*']
    return marks


def get_cmap(style="default"):
    if style == "two_tone":
        return LinearSegmentedColormap.from_list("",["#2851CC", "#9A44C5", "#F96C39"])
    colors = get_col_list()
    colors = [to_rgb(c) for c in colors]
    colors_sorted = [
        c for _, c in sorted(zip([rgb_to_hsv(*c)[0] for c in colors], colors))
    ]
    return LinearSegmentedColormap.from_list("sorted_cmap", colors_sorted, N=256)