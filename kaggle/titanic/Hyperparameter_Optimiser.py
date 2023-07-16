from Basic_NN import Use_Model
from Five_Fold_Stratification import Five_Fold_Stratification
import math
from tqdm import tqdm
import pandas as pd
from itertools import product
from collections import OrderedDict


class Hyperparameter_Optimiser:
    def __init__(self, in_df, output_cols, use_model_args):
        self.df = in_df
        self.output_cols = output_cols
        self.use_model_args = use_model_args
        self.stratifier = Five_Fold_Stratification(in_df)
        self.matrix_data = pd.DataFrame()

    def get_best_params(
        self,
        model_args_range,
        metric="mean_squared_error",
        metric_smaller_better=True,
        stratify=True,
    ):
        model_args_range = OrderedDict(model_args_range)
        # best = {
        #     "stat": math.inf if metric_smaller_better else -math.inf,
        #     "layers": None,
        #     "nodes": None,
        # }
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

            # for n_layers in tqdm(
            #     range(min_layers, max_layers + 1, layer_stride), desc="Layers"
            # ):
            #     for n_nodes in range(min_nodes, max_nodes + 1, node_stride):
            #         tqdm.write(f"\nOptimising: {[n_nodes] * n_layers}")
            #         validation = self._train_test(
            #             metric, n_layers, n_nodes, stratify
            #         )
            #         data["n_layers"].append(n_layers)
            #         data["n_nodes"].append(n_nodes)
            #         data["validation"].append(validation)
            #         if (
            #             metric_smaller_better and validation < best["stat"]
            #         ) or (
            #             not metric_smaller_better and validation > best["stat"]
            #         ):
            #             best["stat"] = validation
            #             best["layers"] = n_layers
            #             best["nodes"] = n_nodes
        self.matrix_data = pd.DataFrame(data)
        return self.matrix_data

    def _train_test(self, metric, model_args, stratify):
        use_model_args = self.use_model_args.copy()
        use_model_args["metrics"] = [metric]
        use_model_args["model_args"] = model_args
        if stratify:
            validation = self.stratifier.verify_model_on_folds(
                output_cols=self.output_cols,
                use_model_args=use_model_args,
            )
        else:
            model = Use_Model(
                self.df.copy(),
                output_cols=self.output_cols,
                **use_model_args,
            )
            model.fit(verbose="silent")
            validation = model.validate(verbose="silent")[1]
        return validation

    # # def plot_result(self, filepath):
    # if self.matrix_data.empty():
    #     #ohdear
    #     print(matrix_data)

    #     plt.scatter(
    #         self.matrix_data["n_layers"],
    #         matrix_data["n_nodes"],
    #         c=matrix_data["validation"],
    #         cmap="jet",
    #     )
    #     plt.xlabel("n_layers")
    #     plt.ylabel("n_nodes")
    #     plt.title("Hyperparameter Matrix Plot")
    #     plt.colorbar(label="Accuracy")
    #     plt.savefig(filepath)  # Save the plot as
