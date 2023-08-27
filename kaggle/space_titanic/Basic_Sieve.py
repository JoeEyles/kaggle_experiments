import pandas as pd
from Basic_NN import Basic_NN
from feature_munger import one_hot_encode_col


class Basic_Sieve:
    """Sieve that just uses if an output is valid or not,
    it does not use anything cleverer
    """

    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, train_x, train_y, model_args):
        train_y, model_args = self._set_up_output_cols(train_y, model_args)
        self._build_sieve(train_x, train_y, model_args)

    def validate(self, validate_x, validate_y):
        validate_y = one_hot_encode_col(validate_y, self.original_output_col)
        prediction_df = self._use_sieve(validate_x)
        validate_y["output"] = (
            validate_y[self.output_cols[0]] > validate_y[self.output_cols[1]]
        )
        prediction_df["output"] = (
            prediction_df[self.output_cols[0]]
            > prediction_df[self.output_cols[1]]
        )
        result = (
            prediction_df["output"] == validate_y["output"]
        )  # TODO: I think the indexes might not match by here...
        return _get_hit_rate(result)

    def _build_sieve(self, train_x, train_y, model_args):
        train_x_remaining = train_x.copy()
        train_y_remaining = train_y.copy()
        print(f"Starting with {len(train_x_remaining)}")
        self.NNs = []
        while len(self.NNs) < self.max_depth:
            basic_nn = Basic_NN(**model_args)
            basic_nn.fit(train_x_remaining, train_y_remaining)
            prediction_df = basic_nn.predict(train_x_remaining)
            prediction_df["valid"] = prediction_df.apply(
                _is_prediction_valid, output_cols=self.output_cols
            )
            train_x_remaining = train_x_remaining[
                prediction_df["valid"] == False
            ]
            train_y_remaining = train_y_remaining[
                prediction_df["valid"] == False
            ]
            print(f"Reduced to {len(train_x_remaining)}")
            if train_x_remaining.empty:
                break
            self.NNs.append(basic_nn)
        print(
            f"Built sieve of  depth {len(self.NNs)}, with remaining data {len(train_x_remaining)}"
        )

    def _use_sieve(self, df_x):
        df_x_remaining = df_x.copy()
        final_prediction_dfs = []
        for NN in self.NNs:
            prediction_df = NN.predict(df_x_remaining)
            prediction_df["valid"] = prediction_df.apply(
                _is_prediction_valid, output_cols=self.output_cols
            )
            df_x_remaining = df_x_remaining[prediction_df["valid"] == False]
            final_prediction_dfs.append(
                prediction_df.merge(prediction_df[prediction_df["valid"]])
            )
        prediction_df = pd.concat(final_prediction_dfs)
        return prediction_df

    def _set_up_output_cols(self, train_y, model_args):
        if len(model_args["output_cols"]) > 1:
            raise ValueError("Sieve not supported for more than 1 feature")
        output_col = model_args["output_cols"][0]
        train_y = one_hot_encode_col(train_y, output_col)
        output_cols = [f"{output_col}_1", f"{output_col}_0"]
        model_args["output_cols"] = output_cols
        self.output_cols = output_cols
        self.original_output_col = output_col
        return train_y, model_args


def _is_prediction_valid(row, output_cols):
    n_greater_0p5 = 0
    for col in output_cols:
        if row[col] > 0.5:
            n_greater_0p5 += 1
    return n_greater_0p5 == 1


def _get_hit_rate(result_col):
    hits = result_col.astype(int).sum()
    total = len(result_col)
    hit_rate = hits / total
    return hit_rate
