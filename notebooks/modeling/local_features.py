import pandas as pd
import numpy as np


def add_relative_dates(data):
    """Shifts the dates relative to 2010 (year = year-2010)"""
    cols = ["YrSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    avg_year = 2010
    for col in cols:
        data[col] = data[col].fillna(data[col].mean()) - avg_year
    years = data[cols]
    years_relative = years[["YrSold"]].values - years.drop("YrSold", axis=1)
    return data.join(years_relative, rsuffix="_Relative")


def add_relative_baths(data):
    baths = data[["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]]
    area = data["GrLivArea"]
    data["baths_relative"] = baths.sum(1) / area
    return data


def month_to_categorical(data):
    data["MoSold"] = pd.Categorical(data["MoSold"])
    return data


def merge_conditions(data):
    cond1 = pd.get_dummies(data["Condition1"], prefix="Condition")
    cond2 = pd.get_dummies(data["Condition2"], prefix="Condition")
    return (
        data.drop(["Condition1", "Condition2"], axis=1)
        .join(cond1.add(cond2, fill_value=0))
        .drop("Condition_Normal", axis=1)
    )


def merge_exterior(data):
    cond1 = pd.get_dummies(data["Exterior1st"], prefix="Exterior")
    cond2 = pd.get_dummies(data["Exterior2nd"], prefix="Exterior")
    data = data.drop(["Exterior1st", "Exterior2nd"], axis=1).join(
        cond1.add(cond2, fill_value=0)
    )
    return data


def add_relative_rooms(data):
    # total rooms above ground including a staircase if the apartment is 2-floor
    data["TotalRoomsAbv"] = data[["FullBath", "HalfBath", "TotRmsAbvGrd"]].sum(1) + (
        data["2ndFlrSF"] > 0
    )
    data["TotalRoomsAbvRel"] = data["TotalRoomsAbv"] / data["GrLivArea"]
    return data


def to_log(data):
    cols = [
        "MiscVal",
        "OpenPorchSF",
        "1stFlrSF",
        "GrLivArea",
        "ScreenPorch",
        "GarageArea",
        "TotalBsmtSF",
        "LotArea",
        "LowQualFinSF",
        "EnclosedPorch",
        "2ndFlrSF",
        "BsmtUnfSF",
        "WoodDeckSF",
        "BsmtFinSF2",
        "LotFrontage",
        "MasVnrArea",
        "PoolArea",
        "3SsnPorch",
        "BsmtFinSF1",
    ]
    for col in cols:
        data[col + "_log"] = np.log1p(data[col])
    return data.drop(cols, axis=1)


def add_features(data, replacements_quant={}, replacements_cat={}):
    """Adds new local features (features based only on the current
    row).

    :param replacements_quant: list of replacements to convert
        categorical columns to numerical ones

    :param replacements_cat: list of replacements for categorical
        columns, typically to manually simplify categories
    """

    data = data.copy()

    for col, reps in replacements_cat.items():
        data[col] = data[col].replace(reps).astype("category")

    for col, reps in replacements_quant.items():
        data[col] = data[col].replace(reps).fillna(0).astype("int")

    data = (
        data.pipe(add_relative_dates)
        .pipe(add_relative_baths)
        .pipe(merge_conditions)
        .pipe(merge_exterior)
        .pipe(month_to_categorical)
        .pipe(add_relative_rooms)
        .pipe(to_log)
        .drop(["Utilities"], axis=1)  # utilities have no diversity at all
    )

    for col in data.select_dtypes(["object", "category"]):
        data[col] = pd.Categorical(data[col].replace(np.nan, "None"))

    for col in data.select_dtypes(exclude="category").drop("SalePrice", axis=1):
        data[col].fillna(0, inplace=True)

    # no nans aside from the SalePrice
    assert data.drop("SalePrice", axis=1).notna().all(axis=None)

    # no object columns
    assert not (data.dtypes == "object").any()

    # no constant columns
    assert (data.std() > 0).all()

    return data


def drop_rare_categorical(series, treashold=50, value="Other"):
    assert isinstance(series, pd.Series)
    assert isinstance(treashold, int)

    if series.dtype.name != "category":
        return series

    replacements = {
        val: value for val, n in series.value_counts().items() if n < treashold
    }

    if len(replacements) > 1:
        series = series.replace(replacements)
    return series
