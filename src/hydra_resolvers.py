################################################################################
#
# Custom resolvers for hydra configuration
#
# Author(s): Nik Vaessen
################################################################################

import uuid

################################################################################
# implement division of 2 digits


def _parse_digit(d: str):
    try:
        d = int(d)
    except ValueError:
        try:
            d = float(d)
        except ValueError:
            raise ValueError(f"input {d} cannot be parsed as a digit")

    return d


def division_resolver(numerator: str, denominator: str):
    return _parse_digit(numerator) / _parse_digit(denominator)


def integer_division_resolver(numerator: str, denominator: str):
    return int(_parse_digit(numerator) // _parse_digit(denominator))


################################################################################
# create a random UUID


def random_uuid():
    return uuid.uuid4().hex
