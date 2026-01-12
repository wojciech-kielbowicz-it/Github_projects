# Methods:

def style_function(feature):
    """
    Defines the default visual style for the GeoJSON features.

    Returns a dictionary configuring a transparent white fill with 
    subtle grey borders, serving as the base state before interaction.
    """
    return {
        "fillColor": "white",
        "color": "#bbbbbb",
        "weight": 0.5,
        "fillOpacity": 0.01
    }

def highlight_function(feature):
    """
    Defines the visual style applied to a feature on mouse hover.

    Returns a dictionary that highlights the specific feature with 
    a red fill, increased opacity, and a bold black border.
    """
    return {
        "fillColor": "red",
        "color": "black",
        "weight": 2,
        "fillOpacity": 0.7
    }