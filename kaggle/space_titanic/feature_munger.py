import pandas as pd
import numpy as np

from models import make_RF_model, make_NN_model_regression_feature_munger


def one_hot_encode_col(df, col):
    df_encoded = pd.get_dummies(df[col], prefix=col)
    df_encoded = pd.concat([df, df_encoded], axis=1)
    return df_encoded


def fill_nans_with_basic_ml_classifier(df, col_name, feature_cols, model=None):
    df_copy = df.copy()
    for col in feature_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    df_train = df_copy.loc[~np.isnan(df_copy[col_name])]
    df_predict = df_copy.loc[np.isnan(df_copy[col_name])].copy()
    if model is None:
        model = make_RF_model(feature_cols, [col_name])
        model.fit(df_train[feature_cols], df_train[col_name])
    df_predict[col_name] = model.predict(df_predict[feature_cols])
    new_df = (
        df.set_index("PassengerId")
        .fillna(df_predict.set_index("PassengerId"))
        .reset_index()
    )
    return new_df, model


def fill_nans_with_basic_ml_regression(df, col_name, feature_cols, model=None):
    df_copy = df.copy()
    for col in feature_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    df_train = df_copy.loc[~np.isnan(df_copy[col_name])]
    df_predict = df_copy.loc[np.isnan(df_copy[col_name])].copy()
    if model is None:
        model = make_NN_model_regression_feature_munger(
            feature_cols, [col_name], []
        )
        model.fit(df_train[feature_cols], df_train[col_name])
    df_predict[col_name] = model.predict(df_predict[feature_cols])
    new_df = (
        df.set_index("PassengerId")
        .fillna(df_predict.set_index("PassengerId"))
        .reset_index()
    )
    return new_df, model


def fill_nans_with_mean_grouped(df, col_name, group_cols, means=None):
    raise IndexError(
        "This shuffles the array into a different order, which causes problems with the submission"
    )
    if means is None:
        means = df.groupby(group_cols)[col_name].mean()
        means.name = "mean"
    df = df.merge(means, on=group_cols)
    df = df.groupby(group_cols).apply(
        fillnans_with_grouped_mean, col_name=col_name
    )
    del df["mean"]
    return df, means


def fillnans_with_grouped_mean(df, col_name):
    mean = df["mean"].iloc[0]
    df[col_name].fillna(mean, inplace=True)
    return df


def fill_nans_with_mean(df, col_name, mean=None):
    if mean is None:
        mean = df[col_name].mean()
    df[col_name].fillna(mean, inplace=True)
    return df, mean


def fill_nans_with_mode(df, col_name, mode=None):
    if mode is None:
        mode = float(df[col_name].mode())
    df[col_name].fillna(mode, inplace=True)
    return df, mode


# def encode_name(df):
#     df["title"] = df["Name"].apply(_get_title)
#     df = one_hot_encode_col(df, "title")
#     return df


# def _get_title(name):
#     if not isinstance(name, str):
#         return np.nan
#     name = name.split(",")[1]
#     name = name.split(".")[0]
#     return name[1:]


# def encode_ticket(df):
#     # TODO: how?
#     pass


# def add_fare_per_ticket_column(df):
#     ticket_counts_dict = df["Ticket"].value_counts()
#     df["number_using_ticket"] = df["Ticket"].apply(
#         _get_ticket_count, ticket_counts_dict=ticket_counts_dict
#     )
#     df["fare_per_ticket"] = df["Fare"] / df["number_using_ticket"]
#     return df


# def _get_ticket_count(ticket, ticket_counts_dict):
#     if not isinstance(ticket, str):
#         return np.nan
#     return ticket_counts_dict[ticket]


# def add_is_child_column(df):
#     df["is_child"] = (df["Age"] <= 16).astype(int)
#     return df


def encode_cabin(df):
    df["Cabin_deck"] = df["Cabin"].apply(_get_cabin_deck)
    df["Cabin_side"] = df["Cabin"].apply(_get_cabin_side)
    df["Cabin_number"] = df["Cabin"].apply(_get_cabin_number)
    df = one_hot_encode_col(df, "Cabin_deck")
    df = one_hot_encode_col(df, "Cabin_side")
    quantiles = df["Cabin_number"].quantile([0.25, 0.5, 0.75])
    df["Cabin_number_quantiles"] = df["Cabin_number"].apply(
        _get_quantile_number, quantiles=quantiles
    )
    df = one_hot_encode_col(df, "Cabin_number_quantiles")
    return df


def _get_quantile_number(n, quantiles):
    if np.isnan(n):
        return np.nan
    for i, q in enumerate(quantiles):
        if n <= q:
            return i
    return len(quantiles)


def _get_cabin_deck(cabin):
    if not isinstance(cabin, str):
        return np.nan
    single_cabin = cabin.split("/")[0]
    return single_cabin


def _get_cabin_side(cabin):
    if not isinstance(cabin, str):
        return np.nan
    single_cabin = cabin.split("/")[2]
    return single_cabin


def _get_cabin_number(cabin):
    if not isinstance(cabin, str):
        return np.nan
    single_cabin = cabin.split("/")[1]
    return int(single_cabin)


def get_passenger_group_size(df):
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["NumWithinGroup"] = df["PassengerId"].str.split("_").str[1]
    groupsize_series = df.groupby("GroupId")["PassengerId"].count()
    groupsize_series.name = "GroupSize"
    df = df.merge(groupsize_series, on="GroupId", how="left")
    return df


def split_row_between_group(df, col):
    mean_series = df.groupby("GroupId")[col].mean()
    mean_series.name = "MeanSpendOverGroup_" + col
    df = df.merge(mean_series, on="GroupId", how="left")
    return df


def get_most_common_ship_location(df):
    df["MostCommonShipLocation"] = df.apply(
        get_most_common_ship_location_row, axis=1
    )
    return df


def get_most_common_ship_location_row(row):
    if row["CryoSleep"]:
        return row["Cabin_side"]  # + row["Cabin_side"]
    spend_cols = [
        "MeanSpendOverGroup_RoomService",
        "MeanSpendOverGroup_ShoppingMall",
        "MeanSpendOverGroup_Spa",
        "MeanSpendOverGroup_VRDeck",
        "MeanSpendOverGroup_FoodCourt",
    ]
    max_spend = 0
    max_col = ""
    for col in spend_cols:
        if row[col] > max_spend:
            max_spend = row[col]
            max_col = col
    if max_spend > row["MeanSpendOverGroup_TotalSpend"] * 0.5:
        return max_col
    return row["Cabin_side"]
