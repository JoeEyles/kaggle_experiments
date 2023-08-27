import pandas as pd
import tensorflow as tf
from models import make_NN_model


class Basic_NN:
    def __init__(
        self,
        output_cols,
        feature_cols,
        cols_to_scale=[],
        model=make_NN_model,
        model_args={},
        seed=42,
        metrics=["accuracy"],
    ):
        self.seed = seed
        tf.keras.utils.set_random_seed(seed)

        self.output_cols = output_cols
        self.feature_cols = feature_cols
        self.cols_to_scale = cols_to_scale

        self.model_args = model_args
        self.metrics = metrics
        self.model = model(
            self.feature_cols,
            self.output_cols,
            metrics,
            seed,
            **model_args,
        )

    def fit(self, train_df_X, train_df_Y, verbose="auto"):
        return self.model.fit(
            train_df_X,
            train_df_Y,
            verbose=verbose,
        )

    def validate(self, validate_df_X=None, validate_df_Y=None, verbose="auto"):
        return self.model.evaluate(
            validate_df_X,
            validate_df_Y,
            verbose=verbose,
        )

    def predict(self, predict_df_X=None, verbose="auto"):
        output = self.model.predict(predict_df_X, verbose=verbose)
        output_df = pd.DataFrame(output, columns=self.output_cols)
        return output_df
