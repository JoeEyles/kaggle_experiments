import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os


def main(in_filepath, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    df = pd.read_csv(in_filepath)
    print(f"Using df:\n{df}")
    plot_histograms(df, out_folder)
    plot_scatters(df, out_folder)


def plot_histograms(df, out_folder):
    plt.figure()
    num_columns = df.shape[1]

    num_rows = int(num_columns**0.5)
    num_cols = int(num_columns / num_rows)
    num_cols += num_columns % num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    axes = axes.flatten()

    for i in range(num_columns):
        try:
            axes[i].hist(df.iloc[:, i], bins=10)
            axes[i].set_title(df.columns[i])
        except Exception:
            pass

    if num_columns < len(axes):
        for j in range(num_columns, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "histograms.png"))


def plot_scatters(df, out_folder):
    plt.figure()
    scatter_matrix(df, figsize=(8, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "scatters.png"))


def print_nan_counts(df):
    print(df.isna().sum())


def print_column_types(df):
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")


if __name__ == "__main__":
    main(
        "kaggle_experiments/kaggle/titanic/data/train.csv",
        "kaggle_experiments/kaggle/titanic/data_exploration",
    )
