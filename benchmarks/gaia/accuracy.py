"""EvoTeam GAIA answer matching."""
import re


def correct(prediction, reference):
    prediction = re.split(r'(?i)final\s*answer\s*:', str(prediction), maxsplit=1)[-1].strip()
    normalize = lambda value: re.sub(r'[^a-z0-9]', '', str(value).lower())
    return normalize(prediction) == normalize(reference)
