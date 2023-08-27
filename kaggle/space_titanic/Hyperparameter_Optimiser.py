from Basic_NN import Basic_NN
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product
from collections import OrderedDict
import matplotlib.pyplot as plt


class Hyperparameter_Optimiser:
    def __init__(
        self,
        data_loader,
        use_model_args,
    ):
        self.use_model_args = use_model_args
        self.data_loader = data_loader
        self.matrix_data = pd.DataFrame()

    def get_best_params(
        self,
        model_args_range,
        metric="mean_squared_error",
        stratify=True,
    ):
        model_args_range = OrderedDict(model_args_range)
        params = model_args_range.keys()
        data = {param: [] for param in params}
        data["validation"] = []
        for param_vals in tqdm(product(*model_args_range.values())):
            model_args = {
                param: param_vals[i] for i, param in enumerate(params)
            }
            tqdm.write(f"\nOptimising: {model_args}")
            validation = self._train_test(metric, model_args, stratify)
            for param in params:
                data[param].append(model_args[param])
            data["validation"].append(validation)
        self.matrix_data = pd.DataFrame(data)
        return self.matrix_data

    def _train_test(self, metric, model_args, stratify):
        use_model_args = self.use_model_args.copy()
        use_model_args["metrics"] = [metric]
        use_model_args["model_args"] = model_args
        validations = []
        for fold in range(5):
            basic_nn = Basic_NN(**use_model_args)
            (
                train_df_X,
                train_df_Y,
                validate_df_X,
                validate_df_Y,
            ) = self.data_loader.get_data_split(  # TODO: passing these in is pretty ugly...
                basic_nn.feature_cols,
                basic_nn.output_cols,
                basic_nn.cols_to_scale,
                fold=fold,
            )
            basic_nn.fit(train_df_X, train_df_Y)
            validation = basic_nn.validate(validate_df_X, validate_df_Y)
            validations.append(validation[1])
        return np.mean(validations)

    def plot_result(self, filepath):
        if self.matrix_data.empty:
            print("Trying to plot empty matrix")
        plt.scatter(
            self.matrix_data["n_layers"],
            self.matrix_data["n_nodes"],
            c=self.matrix_data["validation"],
            cmap="jet",
        )
        plt.xlabel("n_layers")
        plt.ylabel("n_nodes")
        plt.title("Hyperparameter Matrix Plot")
        plt.colorbar(label="Accuracy")
        plt.savefig(filepath)  # Save the plot as
