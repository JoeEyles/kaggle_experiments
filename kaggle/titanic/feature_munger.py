import pandas as pd
import numpy as np


# PassengerId,Pclass,Name,              Sex, Age, SibSp,Parch,Ticket,Fare,  Cabin,Embarked
# 892,        3,     "Kelly, Mr. James",male,34.5,0,    0,    330911,7.8292,B45,  Q


def one_hot_encode_col(df, col):
    df_encoded = pd.get_dummies(df[col], prefix=col)
    df_encoded = pd.concat([df, df_encoded], axis=1)
    return df_encoded


# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# for cabin and name


def fill_nans_with_mean(df, col_name, mean=None):
    if mean is None:
        mean = df[col_name].mean()
    df[col_name].fillna(mean, inplace=True)
    return df, mean


def encode_name(df):
    df["title"] = df["Name"].apply(_get_title)
    df = one_hot_encode_col(df, "title")
    return df


def _get_title(name):
    if not isinstance(name, str):
        return np.nan
    name = name.split(",")[1]
    name = name.split(".")[0]
    return name[1:]


def encode_ticket(df):
    # TODO: how?
    pass


def add_fare_per_ticket_column(df):
    ticket_counts_dict = df["Ticket"].value_counts()
    df["number_using_ticket"] = df["Ticket"].apply(
        _get_ticket_count, ticket_counts_dict=ticket_counts_dict
    )
    df["fare_per_ticket"] = df["Fare"] / df["number_using_ticket"]
    return df


def _get_ticket_count(ticket, ticket_counts_dict):
    if not isinstance(ticket, str):
        return np.nan
    return ticket_counts_dict[ticket]


def add_is_child_column(df):
    df["is_child"] = (df["Age"] <= 16).astype(int)
    return df


def encode_cabin(df):
    df["Cabin_letter"] = df["Cabin"].apply(_get_cabin_letter)
    df["Cabin_number"] = df["Cabin"].apply(_get_cabin_number)
    df = one_hot_encode_col(df, "Cabin_letter")
    return df


def _get_cabin_letter(cabin):
    if not isinstance(cabin, str):
        return np.nan
    single_cabin = cabin.split(" ")[0]
    if len(single_cabin) == 0:
        return np.nan
    return single_cabin[0]


def _get_cabin_number(cabin):
    if not isinstance(cabin, str):
        return cabin
    single_cabin = cabin.split(" ")[0]
    if len(single_cabin) == 0:
        return np.nan
    single_no_letter = single_cabin[1:]
    if len(single_no_letter) == 0:
        return np.nan
    return int(single_no_letter)
