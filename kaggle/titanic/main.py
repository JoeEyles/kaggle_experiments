import pandas as pd
import matplotlib.pyplot as plt
from random import random

from Basic_NN import Use_Model
from models import make_RF_model, make_NN_model
from Hyperparameter_Optimiser import Hyperparameter_Optimiser
from Five_Fold_Stratification import Five_Fold_Stratification
from Feature_Importance import Feature_Importance_Finder
from feature_munger import (
    one_hot_encode_col,
    encode_cabin,
    fill_nans_with_mean,
    add_is_child_column,
    encode_name,
    add_fare_per_ticket_column,
)


def main():
    in_filepath = "kaggle_experiments/kaggle/titanic/data/train.csv"
    df = load_data(in_filepath)

    # TODO: don't bother with this, as we have test.csv
    # df_train_test_validate = df.sample(frac=0.8, random_state=42)
    # df_final_validate = df.drop(df_train_test_validate.index)

    # train_validate_predict(
    #     df, model=make_NN_model, model_args={"n_layers": 5, "n_nodes": 182}
    # )
    # optimise_model(
    #     df,
    #     model=make_NN_model,
    #     model_args_range={"n_layers": [1, 2, 3], "n_nodes": [10, 15, 20]},
    # )
    # feature_importance(
    #     df, model=make_NN_model, model_args={"n_layers": 5, "n_nodes": 182}
    # )
    make_submission(
        df, model=make_NN_model, model_args={"n_layers": 5, "n_nodes": 182}
    )

    # train_validate_predict(
    #     df,
    #     model=make_RF_model,
    #     model_args={"n_estimators": 100, "max_features": None},
    # )
    # optimise_model(
    #     df,
    #     model=make_RF_model,
    #     model_args_range={
    #         "n_estimators": [
    #             50,
    #             100,
    #             500,
    #             1000,
    #             1500,
    #         ],
    #         "max_features": [5, 10, 20, 30, 42],
    #     },
    # )
    feature_importance(
        df,
        model=make_RF_model,
        model_args={"n_estimators": 1000, "max_features": None},
    )


def optimise_model(df, model=make_NN_model, model_args_range={}):
    optimser = Hyperparameter_Optimiser(
        df,
        output_cols=["Survived"],
        use_model_args={
            "fill_cols": [
                {
                    "col": "Age",
                    "method": fill_nans_with_mean,
                    "train_test_value": None,
                },
            ],
            "model": model,
        },
    )
    matrix_data = optimser.get_best_params(
        model_args_range=model_args_range,
        metric="accuracy",
        metric_smaller_better=False,
    )

    # TODO: the below belongs inside the optimiser class
    # print(matrix_data)
    # plt.scatter(
    #     matrix_data["n_layers"],
    #     matrix_data["n_nodes"],
    #     c=matrix_data["validation"],
    #     cmap="jet",
    # )
    # plt.xlabel("n_layers")
    # plt.ylabel("n_nodes")
    # plt.title("Hyperparameter Matrix Plot")
    # plt.colorbar(label="Accuracy")
    # plt.savefig(
    #     "kaggle_experiments/kaggle/titanic/data_exploration/test_hyperparameter_matrix_plot.png"
    # )  # Save the plot as an image file

    # print(f"Best choice is {best_choice}")


def train_validate_predict(df, model=make_NN_model, model_args={}):
    model = Use_Model(
        df,
        output_cols=[
            "Survived",
        ],
        model=model,
        model_args=model_args,
        fill_cols=[
            {
                "col": "Age",
                "method": fill_nans_with_mean,
                "train_test_value": None,
            },
        ],
    )
    fit_history = model.fit()
    validation = model.validate()
    print(validation)
    make_and_evaluate_prediction(model)


def make_submission(train_df, model, model_args):
    in_filepath = "kaggle_experiments/kaggle/titanic/data/test.csv"
    test_df = load_data(in_filepath)
    model = Use_Model(
        train_df,
        output_cols=[
            "Survived",
        ],
        model=model,
        model_args=model_args,
        fill_cols=[
            {
                "col": "Age",
                "method": fill_nans_with_mean,
                "train_test_value": None,
            },
        ],
    )
    fit_history = model.fit()
    validation = model.validate()
    print(f"Training validating at: {validation}")
    prediction_df = model.predict(test_df)
    prediction_df["Survived"] = prediction_df["Survived"].apply(
        lambda x: round(x)
    )
    prediction_df["PassengerId"] = prediction_df["PassengerId"].astype(int)
    submission_df = prediction_df[["PassengerId", "Survived"]]
    # submission_df.to_csv(
    #     "kaggle_experiments/kaggle/titanic/submissions/first_NN.csv",
    #     index=False,
    # )


def feature_importance(df, model=make_NN_model, model_args={}):
    finder = Feature_Importance_Finder(
        df,
        output_cols=["Survived"],
        use_model_args={
            "fill_cols": [
                {
                    "col": "Age",
                    "method": fill_nans_with_mean,
                    "train_test_value": None,
                },
            ],
            # "feature_cols": ["Age", "SibSp", "Parch", "Sex_female"],
        },
    )
    baseline, results_df = finder.get_feaure_importance(
        model_args=model_args,
        model=model,
        stratify=True,
    )
    # TODO: put the below into the Feature_Importance_Finder class
    results_df_sorted = results_df.sort_values("importance", ascending=False)
    plt.bar(results_df_sorted["feature"], results_df_sorted["importance"])
    plt.xlabel("Feature")
    plt.ylabel(f"Importance (difference from baseline of {baseline})")
    plt.title("Feature importance bar chart")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        "kaggle_experiments/kaggle/titanic/data_exploration/test_feature_importance.png"
    )  # Save the plot as an


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = one_hot_encode_col(df, "Sex")
    df = one_hot_encode_col(df, "Embarked")
    df = one_hot_encode_col(df, "Pclass")
    df = encode_cabin(df)
    df = add_is_child_column(df)
    df = encode_name(df)
    df = add_fare_per_ticket_column(df)
    del df["Cabin_number"]  # TODO: how do we fill these blanks?
    return df


def make_and_evaluate_prediction(model):
    prediction_df = model.predict()
    prediction_df["Survived"] = prediction_df["Survived"].apply(
        lambda x: round(x)
    )
    prediction_df["Survived_true"] = (
        model.validate_df["Survived"].reset_index().drop(columns="index")
    )
    # print(prediction_df)
    print(
        f"Hit rate = {get_hit_rate(prediction_df, 'Survived', 'Survived_true')}"
    )
    prediction_df = make_random_guess(prediction_df)
    print(
        f"Hit rate random = {get_hit_rate(prediction_df, 'Survived_random', 'Survived_true')}"
    )
    prediction_df = make_female_guess(prediction_df)
    print(
        f"Hit rate female = {get_hit_rate(prediction_df, 'Survived_female', 'Survived_true')}"
    )


def get_hit_rate(df, col1, col2):
    df["result"] = df[col1] == df[col2]
    hits = df["result"].astype(int).sum()
    total = len(df)
    hit_rate = hits / total
    return hit_rate


def make_random_guess(df):
    df["Survived_random"] = df["Survived"].apply(lambda x: round(random()))
    return df


def make_female_guess(df):
    df["Survived_female"] = df["Sex_female"]
    return df


if __name__ == "__main__":
    main()
