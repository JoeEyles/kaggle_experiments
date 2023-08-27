import pandas as pd
import math

from feature_munger import (
    one_hot_encode_col,
    encode_cabin,
    fill_nans_with_mean,
    add_is_child_column,
    encode_name,
    add_fare_per_ticket_column,
)


class Data_Loader:
    # TODO: currently does not product test_df, only train and validate (since most kaggle competitions have a test seperate)
    def __init__(self, filepath, seed=42):
        self.filepath = filepath
        self.seed = seed
        self.df = _load_data(filepath)
        self.df = _feature_engineering(self.df)

    def get_data_split(self, fold=0):
        train_df, validate_df = _split_data(self.df.copy(), self.seed, fold)
        train_df, validate_df = _fill_gaps(train_df, validate_df)
        return train_df, validate_df


def _split_data(df, seed, fold):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["fold"] = df.index.map(lambda x: math.floor(x % 5))
    train_df = df.loc[df["fold"] == fold].copy()
    validate_df = df.loc[df["fold"] != fold].copy()
    del train_df["fold"]
    del validate_df["fold"]
    return train_df, validate_df


def _load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def _feature_engineering(df):
    df = one_hot_encode_col(df, "Sex")
    df = one_hot_encode_col(df, "Embarked")
    df = one_hot_encode_col(df, "Pclass")
    df = encode_cabin(df)
    df = add_is_child_column(df)
    df = encode_name(df)
    df = add_fare_per_ticket_column(df)
    del df["Cabin_number"]  # TODO: how do we fill these blanks?
    return df


def _fill_gaps(train_df, validate_df):
    train_df, val = fill_nans_with_mean(train_df, "Age", None)
    validate_df, _ = fill_nans_with_mean(train_df, "Age", val)

    train_df, val = fill_nans_with_mean(train_df, "Fare", None)
    validate_df, _ = fill_nans_with_mean(train_df, "Fare", val)

    train_df, val = fill_nans_with_mean(train_df, "fare_per_ticket", None)
    validate_df, _ = fill_nans_with_mean(train_df, "fare_per_ticket", val)
    return train_df, validate_df
