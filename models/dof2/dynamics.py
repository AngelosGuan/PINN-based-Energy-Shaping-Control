import  torch
constants = {
    "m": 5.,
    "m_H": 10.,
    "a": 0.5,
    "b": 0.5,
    "g": 9.8,
    "CONTROL_BOUND": 40.,
}

constants["l"] = constants["a"] + constants["b"]
constants["C_ma"] = (constants["m_H"] + constants["m"]) * constants["l"] * constants["l"] + constants["m"] * constants["a"] * constants["a"]
constants["C_mb"] = - constants["m"] * constants["l"] * constants["b"]
constants["C_mc"] = constants["m"] * constants["b"] * constants["b"]

# state variable ranges
LOWER_BOUNDS = [-0.4, -0.4, -2.0, -2.0]
UPPER_BOUNDS = [ 0.4,  0.4,  2.0,  2.0]
