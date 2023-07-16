from Basic_NN import Use_Model
from Five_Fold_Stratification import Five_Fold_Stratification
import Basic_NN
import pandas as pd


class Feature_Importance_Finder:
    def __init__(self, in_df, use_model_args, output_cols):
        self.df = in_df
        self.use_model_args = use_model_args
        self.output_cols = output_cols
        if "feature_cols" in use_model_args:
            self.feature_cols, _ = Basic_NN.get_feature_cols(
                in_df, output_cols, use_model_args["feature_cols"]
            )
        else:
            self.feature_cols, _ = Basic_NN.get_feature_cols(
                in_df, output_cols
            )
        self.stratifier = Five_Fold_Stratification(in_df)

    def get_feaure_importance(
        self,
        model,
        model_args={},
        stratify=True,
    ):
        results = {"feature": [], "importance": []}
        baseline = self._run_model(
            stratify, self.feature_cols, model_args, model
        )
        for feature in self.feature_cols:
            print(f"Evaluating {feature}")
            new_feature_cols = self.feature_cols.copy()
            new_feature_cols.remove(feature)
            result = self._run_model(
                stratify, new_feature_cols, model_args, model
            )
            results["feature"].append(feature)
            results["importance"].append(baseline - result)
        return baseline, pd.DataFrame(results)

    def _run_model(self, stratify, feature_cols, model_args, model):
        use_model_args = self.use_model_args.copy()
        use_model_args["feature_cols"] = feature_cols
        use_model_args["model_args"] = model_args
        use_model_args["model"] = model
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
