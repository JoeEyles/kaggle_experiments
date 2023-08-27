from random import random
import pandas as pd
from Data_Loader import _expand_cols


class Pipeline:
    def __init__(self, experiment, basic_nns, data_loader):
        self.experiment = experiment
        self.basic_nns = basic_nns
        self.data_loader = data_loader

    def run_experiment(self, nn_idx=0):
        self.experiment.log_Basic_NN_info(self.basic_nns[nn_idx])
        self.experiment.log_Data_Loader_info(self.data_loader)
        (
            train_df_X,
            train_df_Y,
            validate_df_X,
            validate_df_Y,
        ) = self.data_loader.get_data_split(  # TODO: passing these in is pretty ugly...
            self.basic_nns[nn_idx].feature_cols,
            self.basic_nns[nn_idx].output_cols,
            self.basic_nns[nn_idx].cols_to_scale,
        )
        fit_history = self.basic_nns[nn_idx].fit(train_df_X, train_df_Y)
        self.experiment.log_fit(self.basic_nns[nn_idx], fit_history)
        try:
            self._get_rf_feature_importance(train_df_X)  #
        except:
            pass
        validation = self.basic_nns[nn_idx].validate(
            validate_df_X, validate_df_Y
        )
        self.experiment.log_validation(self.basic_nns[nn_idx], validation)
        prediction_df = self.basic_nns[nn_idx].predict(validate_df_X)
        prediction_df = self.data_loader.unscale_data(
            prediction_df,
            self.basic_nns[
                nn_idx
            ].cols_to_scale,  # TODO: passing these in is pretty ugly...
        )
        prediction_df["Transported"] = prediction_df["Transported"].apply(
            lambda x: round(x)
        )
        prediction_df["Transported_true"] = (
            validate_df_Y["Transported"].reset_index().drop(columns="index")
        )
        hit_rate = _get_hit_rate(
            prediction_df, "Transported", "Transported_true"
        )
        print(f"Hit rate = {hit_rate}")
        return prediction_df, hit_rate

    def run_ensemble_experiment(self):
        prediction_df = pd.DataFrame()
        hit_rates = []
        for i in range(len(self.basic_nns)):
            df, hr = self.run_experiment(i)
            hit_rates.append(hr)
            if prediction_df.empty:
                prediction_df["Transported"] = df["Transported"]
                prediction_df["Transported_true"] = df["Transported_true"]
            else:
                prediction_df["Transported"] += df["Transported"]
        prediction_df["Transported"] = (
            prediction_df["Transported"] / len(self.basic_nns)
        ).apply(lambda x: round(x))
        hit_rate = _get_hit_rate(
            prediction_df, "Transported", "Transported_true"
        )
        print(f"Individual hit rates = {hit_rates}")
        print(f"Voted Hit rate = {hit_rate}")
        self.experiment.log_validation(None, hit_rate)

    def run_hyperparameter_optimisation_experiment(self):
        pass


def _make_random_guess(df):
    df["Transported_random"] = df["Transported"].apply(
        lambda x: round(random())
    )
    return df


def _get_hit_rate(df, col1, col2):
    df["result"] = df[col1] == df[col2]
    hits = df["result"].astype(int).sum()
    total = len(df)
    hit_rate = hits / total
    return hit_rate
