import pandas as pd


def add_relative_dates(data):
    cols = ["YrSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    for col in cols:
        data[col] = data[col].fillna(data[col].mean())
    years = data[cols]
    return data.join(
        years[["YrSold"]].values - years.drop("YrSold", axis=1), rsuffix="_Relative"
    )


def add_relative_baths(data):
    baths = data[["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]]
    area = data["GrLivArea"]
    data["baths_relative"] = baths.sum(1) / area
    return data


def month_to_categorical(data):
    data["MoSold"] = pd.Categorical(data["MoSold"])
    return data


def merge_conditions(data):
    cond1 = pd.get_dummies(data["Condition1"], prefix="Condition", dtype=np.bool)
    cond2 = pd.get_dummies(data["Condition2"], prefix="Condition", dtype=np.bool)
    return data.drop(["Condition1", "Condition2"], axis=1).join(cond1 | cond2)


def merge_exterior(data):
    cond1 = pd.get_dummies(data["Exterior1st"], prefix="Exterior", dtype=np.bool)
    cond2 = pd.get_dummies(data["Exterior2nd"], prefix="Exterior", dtype=np.bool)
    return data.drop(["Exterior1st", "Exterior2nd"], axis=1).join(cond1 | cond2)


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
        data[col] = data[col].replace(reps).astype("float").fillna(0)

    # total rooms above ground plus a staircase if the apartment is 2-floor
    data["TotalRoomsAbv"] = data[["FullBath", "HalfBath", "TotRmsAbvGrd"]].sum(1) + (
        data["2ndFlrSF"] > 0
    )
    data["TotalRoomsAbvRel"] = data["TotalRoomsAbv"] / data["GrLivArea"]
    data = (
        data.pipe(add_relative_dates)
        .pipe(add_relative_baths)
        .pipe(merge_conditions)
        .pipe(merge_exterior)
        .pipe(month_to_categorical)
    )

    for col in data.select_dtypes(["object"]).columns:
        data[col] = pd.Categorical(data[col])

    return data
