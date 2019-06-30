import json
from pathlib import Path
import pandas as pd

data_dir = Path(__file__).parent.parent.parent / "data"


def load_data(data_dir=data_dir):
    data = pd.read_hdf(data_dir / "processed" / "data.h5")
    return data


def load_replacements_quant(data_dir=data_dir):
    with open(data_dir / "metadata" / "qualitative.json") as f:
        replacements_quant = json.load(f)
    return replacements_quant


def load_replacements_simp(data_dir=data_dir):
    with open(data_dir / "metadata" / "simplification.json") as f:
        replacements_simp = json.load(f)
    return replacements_simp
