import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from models import make_NN_model


class Use_Model:
    def __init__(
        self,
        df,
        output_cols,
        scale_output=False,
        feature_cols=[],
        scaler_class=StandardScaler,
        model=make_NN_model,
        model_args={},
        seed=42,
        metrics=["accuracy"],
        fill_cols=[],
    ):
        self.output_cols = output_cols
        self.fill_cols = fill_cols

        self.output_cols_to_scale = _guess_cols_to_scale(df, self.output_cols)
        tf.keras.utils.set_random_seed(seed)
        self.train_test_df, self.validate_df = _split_train_test_validate(
            df, seed
        )

        self.feature_cols, self.feature_cols_to_scale = get_feature_cols(
            df, output_cols, feature_cols
        )

        self.model = model(
            self.feature_cols,
            self.output_cols,
            metrics,
            seed,
            **model_args,
        )

        self.feature_scaler = scaler_class()
        if scale_output:
            self.output_scaler = scaler_class()

    def fit(self, in_train_test_df=None, reset_model=True, verbose="auto"):
        if in_train_test_df is None:
            train_test_df = self.train_test_df.copy()
        else:
            train_test_df = in_train_test_df.copy()
        train_test_df.reset_index(inplace=True, drop=True)
        train_test_df, self.fill_cols = _fill_cols(
            train_test_df, self.fill_cols, fit=True
        )
        train_test_df, self.feature_scaler = _scale_df(
            self.feature_scaler,
            train_test_df,
            self.feature_cols_to_scale,
            fit=True,
        )
        if hasattr(self, "output_scaler"):
            train_test_df, self.output_scaler = _scale_df(
                self.output_scaler,
                train_test_df,
                self.output_cols_to_scale,
                fit=True,
            )
        return self.model.fit(
            train_test_df[self.feature_cols],
            train_test_df[self.output_cols],
            verbose=verbose,
        )

    def validate(self, in_validate_df=None, verbose="auto"):
        if in_validate_df is None:
            validate_df = self.validate_df.copy()
        else:
            validate_df = in_validate_df.copy()
        validate_df.reset_index(inplace=True, drop=True)
        validate_df, self.fill_cols = _fill_cols(validate_df, self.fill_cols)
        validate_df, _ = _scale_df(
            self.feature_scaler, validate_df, self.feature_cols_to_scale
        )
        if hasattr(self, "output_scaler"):
            validate_df, _ = _scale_df(
                self.output_scaler, validate_df, self.output_cols_to_scale
            )
        return self.model.evaluate(
            validate_df[self.feature_cols],
            validate_df[self.output_cols],
            verbose=verbose,
        )

    def predict(self, in_predict_df=None, verbose="auto"):
        if in_predict_df is None:
            predict_df = self.validate_df.copy()
        else:
            predict_df = in_predict_df.copy()
        predict_df.reset_index(inplace=True, drop=True)
        predict_df = _add_missing_one_hot_encoded_cols(
            predict_df, self.feature_cols
        )
        predict_df, self.fill_cols = _fill_cols(predict_df, self.fill_cols)
        predict_df, _ = _scale_df(
            self.feature_scaler, predict_df, self.feature_cols_to_scale
        )
        output = self.model.predict(
            predict_df[self.feature_cols], verbose=verbose
        )
        output_df = pd.DataFrame(output, columns=self.output_cols)
        if hasattr(self, "feature_scaler"):
            predict_df = _unscale_df(
                self.feature_scaler,
                predict_df,
                self.feature_cols_to_scale,
            )
        if hasattr(self, "output_scaler"):
            output_df = _unscale_df(
                self.output_scaler,
                output_df,
                self.output_cols_to_scale,
            )
        # output_df = output_df.join(predict_df)
        predict_df[self.output_cols] = output_df[self.output_cols]
        return predict_df


def _add_missing_one_hot_encoded_cols(in_predict_df, feature_cols):
    missing_cols = set(feature_cols) - set(in_predict_df.columns)
    for col in missing_cols:
        in_predict_df[col] = 0
    return in_predict_df


def _split_train_test_validate(df, seed):
    sample_proportion = 0.8
    df_train_test = df.sample(frac=sample_proportion, random_state=seed)
    df_validate = df.drop(df_train_test.index)
    return df_train_test, df_validate


def _scale_df(scaler, df, cols, fit=False):
    if not cols:
        return df, scaler
    scaled_df = df.copy()
    if fit:
        scaler.fit(scaled_df[cols].values)
    scaled_data = scaler.transform(scaled_df[cols].values)
    scaled_data_df = pd.DataFrame(scaled_data, columns=cols)
    for col in cols:
        scaled_df[col] = scaled_data_df[col]
    return scaled_df, scaler


def _unscale_df(scaler, df, cols):
    if not cols:
        return df
    unscaled_data = scaler.inverse_transform(df[cols].values)
    new_df = pd.DataFrame(unscaled_data, columns=cols)
    for col in cols:
        df[col] = new_df[col]
    return df


def _fill_cols(df, fill_cols, fit=False):
    for fill_col in fill_cols:
        col = fill_col["col"]
        method = fill_col["method"]
        val = None
        if not fit:
            val = fill_col["train_test_value"]
        df, val = method(df, col, val)
        fill_col["train_test_value"] = val
    return df, fill_cols


def _guess_cols_to_scale(df, numeric_cols):
    to_scale = []
    for col in numeric_cols:
        if not {0, 1}.issuperset(df[col]):
            to_scale.append(col)
    return to_scale


def get_feature_cols(df, output_cols, feature_cols=[]):
    numeric_input_cols = [
        col
        for col in df.select_dtypes(
            include=[
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "int64",
                "float64",
            ]
        ).columns
        if col not in output_cols
    ]
    if feature_cols:
        feature_cols = [col for col in feature_cols if col not in output_cols]
    else:
        feature_cols = numeric_input_cols
    numeric_feature_cols = [
        col for col in feature_cols if col in numeric_input_cols
    ]
    feature_cols_to_scale = _guess_cols_to_scale(df, numeric_feature_cols)
    return feature_cols, feature_cols_to_scale
