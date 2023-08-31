import pandas as pd
from Basic_NN import Basic_NN
from feature_munger import one_hot_encode_col


class Basic_Sieve:
    """Sieve that just uses if an output is valid or not,
    it does not use anything cleverer.
    You can use the valid_tol to control what is a valid prediction.
    Smaller valid_tol leads to more layers.
    """

    def __init__(self, model_args, max_depth=5, valid_tol=0.5):
        self.max_depth = max_depth
        self.valid_tol = valid_tol
        self.model_args = model_args

    def fit(self, train_x, train_y, verbose=""):
        train_y, model_args = self._set_up_output_cols(train_y)
        del train_y[self.original_output_col]
        self._build_sieve(train_x, train_y, model_args)

    def evaluate(self, validate_x, validate_y, verbose=""):
        validate_y = one_hot_encode_col(validate_y, self.original_output_col)
        prediction_df = self._use_sieve(validate_x)
        acc = self._get_hit_rate(validate_y, prediction_df)
        return [acc, acc]

    def predict(self, predict_x, verbose=""):
        prediction_df = self._use_sieve(predict_x)
        return prediction_df

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
                _is_prediction_valid,
                axis=1,
                output_cols=self.output_cols,
                tol=self.valid_tol,
            )
            train_x_remaining = train_x_remaining[~prediction_df["valid"]]
            train_y_remaining = train_y_remaining[~prediction_df["valid"]]
            print(f"Reduced to {len(train_x_remaining)}")
            self.NNs.append(basic_nn)
            if train_x_remaining.empty:
                break
        print(
            f"Built sieve of depth {len(self.NNs)}, "
            f"with remaining data {len(train_x_remaining)}"
        )

    def _use_sieve(self, df_x):
        df_x_remaining = df_x.copy()
        final_prediction_dfs = []
        for i, NN in enumerate(self.NNs):
            prediction_df = NN.predict(df_x_remaining)
            if i == len(self.NNs) - 1:
                prediction_df["valid"] = True
            else:
                prediction_df["valid"] = prediction_df.apply(
                    _is_prediction_valid,
                    axis=1,
                    output_cols=self.output_cols,
                    tol=self.valid_tol,
                )
            df_x_remaining = df_x_remaining[~prediction_df["valid"]]
            prediction_df = prediction_df[prediction_df["valid"]]
            final_prediction_dfs.append(prediction_df)
        print(f"Used sieve to depth of {i}")
        prediction_df = pd.concat(final_prediction_dfs)
        return prediction_df

    def _set_up_output_cols(self, train_y):
        if len(self.model_args["output_cols"]) > 1:
            raise ValueError("Sieve not supported for more than 1 feature")
        output_col = self.model_args["output_cols"][0]
        train_y = one_hot_encode_col(train_y, output_col)
        output_cols = [f"{output_col}_1", f"{output_col}_0"]
        self.model_args["output_cols"] = output_cols
        self.output_cols = output_cols
        self.original_output_col = output_col
        return train_y, self.model_args

    def _get_hit_rate(self, validate_y, prediction_df):
        validate_y["output_true"] = validate_y[self.original_output_col] == 1
        prediction_df["output_prediction"] = (
            prediction_df[self.output_cols[0]]
            < prediction_df[self.output_cols[1]]
        )
        result_df = prediction_df.join(validate_y, rsuffix="validate")
        for col in self.output_cols:
            del result_df[col]
        result_df["output_result"] = (
            result_df["output_true"] == result_df["output_prediction"]
        )
        hits = result_df["output_result"].astype(int).sum()
        total = len(result_df["output_result"])
        hit_rate = hits / total
        return hit_rate


def _is_prediction_valid(row, output_cols, tol):
    n_greater_0p5 = 0
    for col in output_cols:
        if row[col] > tol:
            n_greater_0p5 += 1
    return n_greater_0p5 == 1
