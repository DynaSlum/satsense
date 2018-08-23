"""
Mappings for satelite image bands

0-index based for python, when using gdal add 1
"""
BANDS = {
    'worldview2': {
        'coastal': 0,
        'blue': 1,
        'green': 2,
        'yellow': 3,
        'red': 4,
        'red-edge': 5,
        'nir-1': 6,
        'nir-2': 7,
    },
    'worldview3': {
        'coastal': 0,
        'blue': 1,
        'green': 2,
        'yellow': 3,
        'red': 4,
        'red-edge': 5,
        'nir-1': 6,
        'nir-2': 7,
    },
    'quickbird': {
        'blue': 0,
        'green': 1,
        'red': 2,
        'nir-1': 3,
    },
    'rgb': {
        'red': 0,
        'green': 1,
        'blue': 2,
    },
    'monochrome': {
        'pan': 0,
    },
    'pleiades': {
        'pan': 0,
        'blue': 1,
        'green': 2,
        'red': 3,
        'nir-1': 4,
    }
}
