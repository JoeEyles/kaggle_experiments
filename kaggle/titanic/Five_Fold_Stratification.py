import math
from Basic_NN import Use_Model


class Five_Fold_Stratification:
    def __init__(
        self,
        df,
    ):
        self.df = df

    def verify_model_on_folds(self, use_model_args, output_cols):
        df = self.df.copy().sample(frac=1).reset_index(drop=True)
        df["fold"] = df.index.map(lambda x: math.floor(x % 5))
        score_sum = 0
        for fold in range(0, 5):
            df_validate = df.loc[df["fold"] == fold].copy()
            df_train_test = df.loc[df["fold"] != fold].copy()
            del df_validate["fold"]
            del df_train_test["fold"]
            model = Use_Model(
                df_train_test, output_cols=output_cols, **use_model_args
            )
            model.fit(df_train_test, verbose="silent")
            validation = model.validate(df_validate, verbose="silent")
            score_sum += validation[1]
        return score_sum / 5
