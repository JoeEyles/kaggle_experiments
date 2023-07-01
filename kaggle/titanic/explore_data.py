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
    columns = df.select_dtypes(include=["int64", "float64"]).columns
    df = df[columns].copy()
    num_columns = df.shape[1]  # Get the number of columns in the data

    # Calculate the number of rows and columns for subplots
    num_rows = int(num_columns**0.5)
    num_cols = int(num_columns / num_rows)
    num_cols += num_columns % num_rows  # Add an extra column if needed

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    for i in range(num_columns):
        axes[i].hist(df.iloc[:, i], bins=10)
        axes[i].set_title(df.columns[i])

    if num_columns < len(axes):
        for j in range(num_columns, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "histograms.png"))


def plot_scatters(df, out_folder):
    plt.figure()
    columns = df.select_dtypes(include=["int64", "float64"]).columns
    df = df[columns].copy()
    scatter_matrix(df, figsize=(8, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "scatters.png"))


if __name__ == "__main__":
    main(
        "kaggle_experiments/kaggle/titanic/data/train.csv",
        "kaggle_experiments/kaggle/titanic/data_exploration",
    )
