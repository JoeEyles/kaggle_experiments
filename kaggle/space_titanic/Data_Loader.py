import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

from feature_munger import (
    one_hot_encode_col,
    encode_cabin,
    fill_nans_with_mean,
    fill_nans_with_basic_ml_classifier,
    fill_nans_with_basic_ml_regression,
    fill_nans_with_mean_grouped,
    fill_nans_with_mode,
    get_passenger_group_size,
    split_row_between_group,
    get_most_common_ship_location,
    #     add_is_child_column,
    #     encode_name,
    #     add_fare_per_ticket_column,
)


class Data_Loader:
    # TODO: currently does not product test_df, only train and validate (since most kaggle competitions have a test seperate)
    def __init__(
        self,
        filepath,
        seed=42,
        scaler_class=StandardScaler,
    ):
        self.filepath = filepath
        self.seed = seed
        self.scaler = scaler_class()
        self.df = _load_data(filepath)

    def get_data_split(
        self,
        feature_cols,
        output_cols,
        cols_to_scale=[],
        fold=0,
    ):
        train_df, validate_df = _split_data(self.df.copy(), self.seed, fold)
        train_df, validate_df = _fill_gaps_simple_ml(train_df, validate_df)
        train_df, validate_df = _feature_engineering(train_df, validate_df)
        validate_df = _add_missing_one_hot_encoded_cols(
            validate_df, _expand_cols(feature_cols, train_df)
        )
        train_df = self._scale_df(
            train_df, _expand_cols(cols_to_scale, train_df), fit=True
        )
        validate_df = self._scale_df(
            validate_df,
            _expand_cols(cols_to_scale, validate_df),
            fit=False,
        )
        train_df_X = self._select_expanded_cols_from_df(train_df, feature_cols)
        train_df_Y = self._select_expanded_cols_from_df(train_df, output_cols)
        validate_df_X = self._select_expanded_cols_from_df(
            validate_df, feature_cols
        )
        validate_df_Y = self._select_expanded_cols_from_df(
            validate_df, output_cols
        )
        validate_df_X = validate_df_X.reindex(train_df_X.columns, axis=1)
        validate_df_Y = validate_df_Y.reindex(train_df_Y.columns, axis=1)
        return train_df_X, train_df_Y, validate_df_X, validate_df_Y

    def unscale_data(self, df, cols_to_scale):
        df = self._unscale_df(df, _expand_cols(cols_to_scale, df))
        return df

    def _select_expanded_cols_from_df(self, df, col_list):
        expanded_col_list = _expand_cols(col_list, df)
        return df[expanded_col_list]

    def _scale_df(self, df, cols, fit=False):
        if not cols:
            return df
        scaled_df = df.copy()
        scaled_df.reset_index(drop=True, inplace=True)
        if fit:
            self.scaler.fit(scaled_df[cols].values)
        scaled_data = self.scaler.transform(scaled_df[cols].values)
        scaled_data_df = pd.DataFrame(scaled_data, columns=cols)
        for col in cols:
            scaled_df[col] = scaled_data_df[col]
        return scaled_df

    def _unscale_df(self, df, cols):
        if not cols:
            return df
        unscaled_data = self.scaler.inverse_transform(df[cols].values)
        new_df = pd.DataFrame(unscaled_data, columns=cols)
        for col in cols:
            df[col] = new_df[col]
        return df


def _expand_cols(col_list, df):
    df_cols = df.columns.tolist()
    chosen_cols = []
    for col in col_list:
        if col in df_cols:
            chosen_cols.append(col)
        if col.endswith("*"):
            col_prefix = col[:-1]
            for df_col in df_cols:
                if df_col.startswith(col_prefix) and len(df_col) > len(
                    col_prefix
                ):
                    chosen_cols.append(df_col)
    return chosen_cols


def _split_data(df, seed, fold):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["fold"] = df.index.map(lambda x: math.floor(x % 5))
    train_df = df.loc[df["fold"] != fold].copy()
    validate_df = df.loc[df["fold"] == fold].copy()
    del train_df["fold"]
    del validate_df["fold"]
    return train_df, validate_df


def _load_data(filepath):
    df = pd.read_csv(filepath)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x) if type(x) == bool else x)
    return df


# def _fill_gaps_basic_mean_mode(train_df, validate_df):
#     for col in [
#         "CryoSleep",
#         "VIP",
#     ]:
#         train_df, mode = fill_nans_with_mode(train_df, col)
#         validate_df, _ = fill_nans_with_mode(validate_df, col, mode)

#     for col in [
#         "Age",
#         "RoomService",
#         "ShoppingMall",
#         "Spa",
#         "VRDeck",
#         "FoodCourt",
#     ]:
#         train_df, model = fill_nans_with_mean(train_df, col)
#         validate_df, _ = fill_nans_with_mean(validate_df, col, model)

#     return train_df, validate_df


def _fill_gaps_simple_ml(train_df, validate_df):
    fill_features = [
        "Age",
        "RoomService",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "FoodCourt",
    ]
    for col in [
        "CryoSleep",
        "VIP",
    ]:
        train_df, mode = fill_nans_with_basic_ml_classifier(
            train_df, col, fill_features
        )
        validate_df, _ = fill_nans_with_basic_ml_classifier(
            validate_df, col, fill_features, mode
        )
    for col in [
        "Age",
        "RoomService",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "FoodCourt",
    ]:
        train_df, means = fill_nans_with_mean_grouped(
            train_df,
            col,
            ["VIP", "CryoSleep"],
        )
        validate_df, _ = fill_nans_with_mean_grouped(
            validate_df,
            col,
            ["VIP", "CryoSleep"],
            means,
        )
    return train_df, validate_df


def _feature_engineering(train_df, validate_df):
    train_df = encode_cabin(train_df)
    validate_df = encode_cabin(validate_df)

    train_df = get_passenger_group_size(train_df)
    validate_df = get_passenger_group_size(validate_df)

    for col in ["HomePlanet", "Destination", "GroupSize"]:
        train_df = one_hot_encode_col(train_df, col)
        validate_df = one_hot_encode_col(validate_df, col)

    train_df["TotalSpend"] = (
        train_df["RoomService"]
        + train_df["FoodCourt"]
        + train_df["ShoppingMall"]
        + train_df["Spa"]
        + train_df["VRDeck"]
    )
    validate_df["TotalSpend"] = (
        validate_df["RoomService"]
        + validate_df["FoodCourt"]
        + validate_df["ShoppingMall"]
        + validate_df["Spa"]
        + validate_df["VRDeck"]
    )

    for col in [
        "RoomService",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "FoodCourt",
        "TotalSpend",
    ]:
        train_df = split_row_between_group(train_df, col)
        validate_df = split_row_between_group(validate_df, col)

    train_df = get_most_common_ship_location(train_df)
    validate_df = get_most_common_ship_location(validate_df)

    train_df = one_hot_encode_col(train_df, "MostCommonShipLocation")
    validate_df = one_hot_encode_col(validate_df, "MostCommonShipLocation")
    return train_df, validate_df


def _add_missing_one_hot_encoded_cols(df, feature_cols):
    missing_cols = set(feature_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    return df
