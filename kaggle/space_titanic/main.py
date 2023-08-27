import os

from Data_Loader import Data_Loader
from Experiment_Manager import Experiment
from Experiment_Pipeline import Pipeline
from Basic_NN import Basic_NN
from Hyperparameter_Optimiser import Hyperparameter_Optimiser
from Basic_Sieve import Basic_Sieve
from models import make_RF_model, make_SVM_model, make_NN_model
from explore_data import (
    plot_histograms,
    plot_scatters,
    print_nan_counts,
    print_column_types,
)


base_path = "kaggle_experiments/kaggle/space_titanic"

feature_cols = [
    "CryoSleep",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "HomePlanet*",
    "Destination*",
    "Cabin_deck*",
    "Cabin_side*",
    "Cabin_number_quantiles*",
    "GroupSize*",
    "MeanSpendOverGroup_*",
    "MostCommonShipLocation*",
]
cols_to_scale = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
]


def main():
    # explore_data()
    # experiment()
    # ensemble_experiment()
    # hyperparameter_optimiser()
    sieve_test()


def experiment():
    data_loader = Data_Loader(os.path.join(base_path, "data/train.csv"))
    basic_nn = Basic_NN(
        model=make_SVM_model,  # make_RF_model,
        model_args={
            # "n_estimators": 1000,
            "C": 1.0,
        },
        output_cols=["Transported"],
        feature_cols=[
            "CryoSleep",
            "Age",
            "VIP",  #
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "TotalSpend",
            "HomePlanet*",
            "Destination*",
            "Cabin_deck*",  #
            # "Cabin_deck_E",
            # "Cabin_deck_G",
            # "Cabin_deck_F",
            "Cabin_side*",
            "Cabin_number_quantiles*",
            "GroupSize*",  #
            "MeanSpendOverGroup_*",
            "MostCommonShipLocation*",  #
            # "MostCommonShipLocation_MeanSpendOverGroup__FoodCourt"
            # "MostCommonShipLocation_MeanSpendOverGroup__ShoppingMall"
            # -------------------
            # "CryoSleep",
            # "Age",
            # "VIP",
            # "RoomService",
            # "FoodCourt",
            # "ShoppingMall",
            # "Spa",
            # "VRDeck",
            # "TotalSpend",
            # "HomePlanet*",
            # "Destination*",
            # "Cabin_deck*",
            # "Cabin_side*",
            # "Cabin_number_quantiles*",
            # "GroupSize*",
            # "MeanSpendOverGroup_*",
            # "MostCommonShipLocation*",
        ],
        cols_to_scale=[
            "Age",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "TotalSpend",
        ],
    )
    experiment = Experiment(
        name="SVM, linear c10",
        description="""
All features, first SVM attempt using a linear, with dual=False, and setting C=10
    """,
    )
    pipeline = Pipeline(experiment, [basic_nn], data_loader)
    pipeline.run_experiment()


def ensemble_experiment():
    data_loader = Data_Loader(os.path.join(base_path, "data/train.csv"))
    basic_nns = [
        Basic_NN(
            model=make_SVM_model,  # make_RF_model,
            model_args={
                # "n_estimators": 1000,
                "C": 1.0,
            },
            output_cols=["Transported"],
            feature_cols=feature_cols,
            cols_to_scale=cols_to_scale,
        ),
        Basic_NN(
            model=make_RF_model,
            model_args={
                "n_estimators": 1000,
            },
            output_cols=["Transported"],
            feature_cols=feature_cols,
            cols_to_scale=cols_to_scale,
        ),
        # Basic_NN(
        #     model=make_NN_model,
        #     model_args={
        #         "n_layers": 9,
        #         "n_nodes": 200,
        #     },
        #     output_cols=["Transported"],
        #     feature_cols=feature_cols,
        #     cols_to_scale=cols_to_scale,
        # ),
    ]
    experiment = Experiment(
        name="First ensemble, tanh NN",
        description="""
Ensemble, basic voting (mean then round) for SVM, and RF, and NN that has been optimised
And this time the final NN layer is using a tanh
    """,
    )
    pipeline = Pipeline(experiment, basic_nns, data_loader)
    pipeline.run_ensemble_experiment()


def hyperparameter_optimiser():
    data_loader = Data_Loader(os.path.join(base_path, "data/train.csv"))
    optimiser = Hyperparameter_Optimiser(
        data_loader,
        use_model_args={
            "model": make_NN_model,
            "model_args": {},
            "output_cols": ["Transported"],
            "feature_cols": feature_cols,
            "cols_to_scale": cols_to_scale,
        },
    )
    matrix_data = optimiser.get_best_params(
        model_args_range={
            "n_layers": [6, 7, 8, 9, 10, 11, 12],
            "n_nodes": [150, 175, 200, 225, 250, 275, 300],
        },
        metric="accuracy",
    )
    print(matrix_data)
    optimiser.plot_result(
        os.path.join(base_path, "optimiser/NN_refined1_search.png")
    )


def explore_data():
    data_loader = Data_Loader(os.path.join(base_path, "data/train.csv"))
    (
        train_df_X,
        train_df_Y,
        validate_df_X,
        validate_df_Y,
    ) = data_loader.get_data_split(feature_cols="*", output_cols="*")
    print_nan_counts(train_df_X)
    print_column_types(train_df_X)
    plot_histograms(train_df_X, os.path.join(base_path, "data_exploration"))
    plot_scatters(train_df_X, os.path.join(base_path, "data_exploration"))


def sieve_test():
    sieve = Basic_Sieve(max_depth=10)
    data_loader = Data_Loader(os.path.join(base_path, "data/train.csv"))
    (
        train_df_X,
        train_df_Y,
        validate_df_X,
        validate_df_Y,
    ) = data_loader.get_data_split(
        feature_cols=feature_cols,
        output_cols=["Transported"],
        cols_to_scale=cols_to_scale,
    )
    sieve.fit(
        train_df_X,
        train_df_Y,
        {
            "model": make_RF_model,
            "model_args": {
                "n_estimators": 1000,
            },
            "output_cols": ["Transported"],
            "feature_cols": feature_cols,
            "cols_to_scale": cols_to_scale,
        },
    )


if __name__ == "__main__":
    main()
