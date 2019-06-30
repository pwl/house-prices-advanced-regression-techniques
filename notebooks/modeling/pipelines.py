class PreprocessingSimple():
    def __init__(self, data):
preprocessing = {}

preprocessing["simple"] = make_pipeline(
    DataFrameMapper(
        [
            (categoricals, OneHotEncoder(), {"alias": "cat"}),
            (categoricals, [TargetEncoder(), impute], {"alias": "cat_target"}),
            (numerical, [impute, StandardScaler()], {"alias": "num"}),
            (
                numerical,
                [impute, StandardScaler(), umap.UMAP(n_components=2)],
                {"alias": "num_UMAP"},
            ),
            (
                numerical,
                [impute, StandardScaler(), KernelPCA(kernel="linear", n_components=5)],
                {"alias": "num_kPCA_lin"},
            ),
        ],
        input_df=True,
        df_out=True,
        default=None,
    ),
    tofloat,
)
