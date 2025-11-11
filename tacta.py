HSVCartesian = tuple[float, float, float]

def hsv_to_cartesian(h: int, s: int, v: int) -> HSVCartesian:
    """Convert HSV color to Cartesian coordinates.
    
    HSV should have values in the range:
    - h: 0-179
    - s: 0-255
    - v: 0-255

    Outputs x, y, z coordinates in the range:
    - x: 0 to 2 pi radians
    - y: 0 to 1
    - z: 0 to 1
    """
    import math

    # Convert hue from [0, 179] to [0, 2*pi]
    x = (h / 180) * 2 * math.pi

    # Convert saturation from [0, 255] to [0, 1]
    y = s / 255

    # Convert value from [0, 255] to [0, 1]
    z = v / 255

    return x, y, z

def cartesian_to_hsv(x: float, y: float, z: float) -> tuple[int, int, int]:
    """Convert Cartesian coordinates back to HSV color.
    
    Inputs x, y, z coordinates in the range:
    - x: 0 to 2 pi radians
    - y: 0 to 1
    - z: 0 to 1

    Outputs HSV should have values in the range:
    - h: 0-179
    - s: 0-255
    - v: 0-255
    """
    import math

    # Convert x from [0, 2*pi] to [0, 179]
    h = int((x / (2 * math.pi)) * 180) % 180

    # Convert y from [0, 1] to [0, 255]
    s = int(y * 255)

    # Convert z from [0, 1] to [0, 255]
    v = int(z * 255)

    return h, s, v