import os
import matplotlib.pyplot as plt
from shutil import rmtree


base_folder = "C:/Users/joeey/OneDrive/Documents/ML/kaggle_experiments/kaggle/space_titanic/experiments"


class Experiment:
    def __init__(
        self,
        name,
        description,
    ):
        self.exp_number = _get_next_experiment_number(base_folder)
        self.experiment_path = os.path.join(
            base_folder, f"{self.exp_number} {name}"
        )
        if os.path.exists(self.experiment_path):
            print(f"Deleting {self.experiment_path}?")
            input("Press enter to continue...")
            rmtree(self.experiment_path)
        os.makedirs(self.experiment_path, exist_ok=True)
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write(f"# Experiment {name}\n")
            f.write("\n## Description\n")
            f.write(description + "\n")

    def get_folder(self, folder_name):
        path = os.path.join(self.experiment_path, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    def write_readme(self, content):
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write(content)

    def log_Basic_NN_info(self, basic_nn):
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write("\n## Basic_NN info\n")
            f.write("output cols: " + str(basic_nn.output_cols) + "\n")
            f.write("feature cols: " + str(basic_nn.feature_cols) + "\n")
            f.write("cols to scale: " + str(basic_nn.cols_to_scale) + "\n")
            f.write("model args: " + str(basic_nn.model_args) + "\n")
            f.write("model: " + str(basic_nn.model) + "\n")
            f.write("seed: " + str(basic_nn.seed) + "\n")
            f.write("metrics: " + str(basic_nn.metrics) + "\n")

    def log_Data_Loader_info(self, data_loader):
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write("\n## DataLoader info\n")
            f.write("df: " + str(data_loader.df) + "\n")
            f.write("seed: " + str(data_loader.seed) + "\n")
            f.write("scaler class: " + str(data_loader.scaler) + "\n")

    def log_fit(self, basic_nn, history):
        folder = self.get_folder("fit_history")
        # summarize history for metric
        metric = basic_nn.metrics[0]
        try:
            plt.plot(history.history[metric])
            plt.title(f"model {metric}")
            plt.ylabel(metric)
            plt.xlabel("epoch")
            plt.savefig(os.path.join(folder, "metric.png"))
            plt.close()
            # summarize history for loss
            plt.plot(history.history["loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.savefig(os.path.join(folder, "loss.png"))
            plt.close()
        except Exception:
            pass

    def log_validation(self, basic_nn, validation):
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write("\n## Validation info\n")
            if basic_nn is not None:
                f.write("metrics: " + str(basic_nn.metrics) + "\n")
            f.write("validation: " + str(validation) + "\n")

    def log_feature_importance(self, results_df_sorted):
        plt.bar(results_df_sorted["feature"], results_df_sorted["importance"])
        plt.xlabel("Feature")
        plt.title("Feature importance bar chart")
        plt.xticks(rotation=90)
        # plt.rcParams["figure.figsize"] = (20, 30)
        plt.subplots_adjust(bottom=0.5)
        # plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_path, "importance_rf.png"))
        with open(os.path.join(self.experiment_path, "README.md"), "a") as f:
            f.write("\n## Feature importance\n")
            f.write("Feature importances:\n" + str(results_df_sorted) + "\n")


def _get_next_experiment_number(base_folder):
    max_num = -1
    for folder in os.listdir(base_folder):
        num = int(float(folder[:2]))
        max_num = max(max_num, num)
    return f"{max_num+1:.2f}"
