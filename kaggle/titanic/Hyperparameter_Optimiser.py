from Basic_NN import Use_Model
from Five_Fold_Stratification import Five_Fold_Stratification
import math
from tqdm import tqdm
import pandas as pd


class Hyperparameter_Optimiser:
    def __init__(self, in_df, output_cols, model_args):
        self.df = in_df
        self.output_cols = output_cols
        self.model_args = model_args
        self.stratifier = Five_Fold_Stratification(in_df)

    def get_best_params(
        self,
        min_layers,
        max_layers,
        layer_stride,
        min_nodes,
        max_nodes,
        node_stride,
        metric="mean_squared_error",
        metric_smaller_better=True,
        stratify=True,
    ):
        best = {
            "stat": math.inf if metric_smaller_better else -math.inf,
            "layers": None,
            "nodes": None,
        }
        data = {
            "n_layers": [],
            "n_nodes": [],
            "validation": [],
        }
        for n_layers in tqdm(
            range(min_layers, max_layers + 1, layer_stride), desc="Layers"
        ):
            for n_nodes in range(min_nodes, max_nodes + 1, node_stride):
                tqdm.write(f"\nOptimising: {[n_nodes] * n_layers}")
                validation = self._train_test(
                    metric, n_layers, n_nodes, stratify
                )
                data["n_layers"].append(n_layers)
                data["n_nodes"].append(n_nodes)
                data["validation"].append(validation)
                if (metric_smaller_better and validation < best["stat"]) or (
                    not metric_smaller_better and validation > best["stat"]
                ):
                    best["stat"] = validation
                    best["layers"] = n_layers
                    best["nodes"] = n_nodes
        return best, pd.DataFrame(data)

    def _train_test(self, metric, n_layers, n_nodes, stratify):
        model_args = self.model_args.copy()
        model_args["hidden_layers_shape"] = [n_nodes] * n_layers
        model_args["metrics"] = [metric]
        if stratify:
            validation = self.stratifier.verify_model_on_folds(
                output_cols=self.output_cols,
                model_args=model_args,
            )
        else:
            model = Use_Model(
                self.df.copy(),
                output_cols=self.output_cols,
                **model_args,
            )
            model.fit(verbose="silent")
            validation = model.validate(verbose="silent")[1]
        return validation
