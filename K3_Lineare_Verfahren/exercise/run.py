import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def closed_form_solution(X: np.ndarray, y: np.ndarray):
    """Returns the optimal model parameters THETA for the feature matrix X and the real values y.
    X is of shape (N, M) where N denotes the number of data points and M the number of features (+1 for bias).
    y is of shape (N, 1)."""

    "print X: [1.         0.24230205] ... die 1 ist nur ein DUmmy Feature, damit wir weiter mit BIas Term auch rechnen können"
    "[1.         0.24230205] .. Zeile ist eine Zeile wie in Matrix und Spalte ist jeweils ein Feature (0.24230205 ein Messwert vom zweiten Feature)"
    return np.zeros((X.shape[1], 1))


def mse(y_prediction: np.ndarray, y: np.ndarray):
    """Returns the mean squared error for the predictions y_prediction and the real values y.
    y_prediction and y is of shape (N, 1)."""
    "Both are the same length"
    # print("MSE AUFGABE Predictions: ", len(
    #     y_prediction), "MSE AUFGABE real value", len(y))

    sum = 0
    for i in range(len(y_prediction)):
        sum += (y[i] - y_prediction[i])**2

    sum = sum / len(y_prediction)
    print("yoooo type", type(sum))
    # checken, ob sum nicht array ist, da man float erwartet: float64 after /len and int32.numpy bevore
    return sum


def question_1():
    """Returns the correct answer to question 1."""
    # Gucken Sie sich die Scatter-Plots der features [2, 3, 4, 5, 9] an
    # Welches Feature lässt sich am besten mit einem linearen Modell darstellen
    # Antwort aus INDUS CHAS NOX RM TAX
    return "RM"


def question_2():
    """Returns the correct answer to question 2."""
    # Welches Feature lässt sich am schlechtesten mit einem linearen Modell darstellen
    # Antwort aus INDUS CHAS NOX RM TAX
    return "CHAS"


def question_3():
    """Returns the correct answer to question 3."""
    # Gucken sie sich das Modell für alle Features an
    # Welches Feature hat den größten Einfluss
    # Antwort aus INDUS CHAS NOX RM TAX
    return ""


def question_4():
    """Returns the correct answer to question 4."""
    # Welches Feature hat den geringsten Einfluss
    # Antwort aus INDUS CHAS NOX RM TAX
    return ""


def min_max_scaling(X):
    """Applies min-max-scaling to the matrix X.
    Every column is scaled to the interval (0, 1)."""
    return (X - X.min(0)) / (X.max(0) - X.min(0))


def get_linear_regression_training_set_from_df(df: pd.DataFrame, columns: list):
    """Returns the training set X and the target column y from the data frame df.
    columns is a list of integers indicating which features should be included in X.
    y is the target column (must be the last column of the data frame).
    The first column of X is filled with 1.
    The shape of X is (N, M) where N denotes the number of data points
    and M the number of features + 1 (added bias term)."""
    # extract feature columns from data
    X = df.iloc[:, columns].to_numpy()
    # apply scaling to data set so we can better compare different features
    X = min_max_scaling(X)
    # For linear regression we need to add an additional feature which is always 1 for the bias term
    # np.c_ concatenates two 2d arrays along the second axis, which can be seen as adding additional columns
    # np.ones(shape) creates an array of shape shape filled with 1
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # extract target column from data; note that y needs to be of shape (n, 1), i.e. a matrix with a single column
    # this is called a column vector, while a matrix of shape (1, n) is called a row vector
    y = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    return X, y


def plot_features_from_df(df: pd.DataFrame, columns: list, n_plot_columns: int = 2):
    """Plots the features of df indicated by columns.
    columns is a list of integers indicating which features should be plotted.
    y-Axis will always be the target column (must be the last column of the data frame)"""
    n_ax_columns = n_plot_columns
    n_ax_rows = int(len(columns) / n_ax_columns)
    n_ax_rows += 1 if len(columns) % n_ax_columns > 0 else 0
    fig, ax = plt.subplots(n_ax_rows, n_ax_columns, figsize=(
        3 * n_ax_columns, 3 * n_ax_rows), sharey=True)
    for column, ax_ in zip(columns, ax.flat):
        ax_.scatter(df.iloc[:, column], df.iloc[:, -1])
        ax_.set_title(f"{df.columns[column]} ({column})")
    if len(columns) % n_ax_columns > 0:
        for i in range(n_ax_columns - len(columns) % n_ax_columns):
            ax.flat[-(i + 1)].set_axis_off()
    fig.tight_layout()
    fig.canvas.set_window_title('Features')
    plt.show()


def get_predictions(theta, X):
    """Returns the predictions of the linear model theta for the data set X.
    theta is of shape (M, 1), where M denotes the number of model parameters (i.e. number of features + 1 [bias]).
    X is of shape (N, M)."""
    return X.dot(theta)


if __name__ == "__main__":
    # import data set
    df = pd.read_csv("../data/boston_house_prices.csv")
    features = [2, 3, 4, 5, 9]

    # plot features
    plot_features_from_df(df, features, 2)

    # MSE of single features
    print("MSE single features:")
    for i in features:
        X, y = get_linear_regression_training_set_from_df(df, [i])
        theta = closed_form_solution(X, y)
        print(f"{df.columns[i]}: {mse(get_predictions(theta, X), y): 0.2f}")
    print("\n")
    X, y = get_linear_regression_training_set_from_df(df, features)
    theta = closed_form_solution(X, y)
    print("Model:")
    print(" +\n\t".join(
        [f"{np.round(coef, 2)} * ({feature_name})"
         for feature_name, coef in zip(["Bias"] + list(df.columns[features]), theta.flatten())]
    ))
    print(f"MSE Model: {mse(get_predictions(theta, X), y): 0.2f}")
