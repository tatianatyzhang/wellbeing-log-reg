import numpy as np
import math
import csv
import pandas as pd

N_STEPS = 100
STEP_SIZE = 0.001

def load(filename):
    df = pd.read_csv(filename)
    if "Demographic" in df.columns:
        df = df.drop(columns=["Demographic"])
    return df


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def transpose(thetas, xs):
    """
    This function defines what transpose means (to multiply theta_i by x_i and sum them all up).
    """
    sum = 0
    for i in range(len(xs)):
        sum += (thetas[i] * xs[i])

    return sum

def gradient_ascent(df, y_column):
    """
    This function uses gradient ascent to find the optimal theta values to converge on.
    """

    # Initialize an array of length k for the parameters, where k represents the number of parameters there are
    num_params = len(df.columns)
    thetas = [0] * num_params

    for n in range(N_STEPS):
        # Initialize an array of length k for the gradients, where k represents the number of parameters there are
        gradients = [0] * num_params

        # Loop through every training example
        for index, row in df.iterrows():
            xs = [1] + row[:-1].tolist()

            # Loop through every parameter j
            for j in range(num_params):
                gradients[j] += xs[j] * (row[y_column] - sigmoid(transpose(thetas, xs)))

        for j in range(num_params):
            thetas[j] += STEP_SIZE * gradients[j]

    return thetas


def compute_accuracy(df, thetas):
    """
    With the optimal parameters determined, we can use them to find the optimal theta and make our predictions.
    """

    # Split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="stress_level")
    y_test = df["stress_level"]

    accuracy = 0
    num_correct = 0
    total = len(y_test)

    # Loop through every training example and make predictions
    for i, row in X_test.iterrows():
        xs = [1] + row.tolist()
        if sigmoid(transpose(thetas, xs)) > 0.5:
                num_correct += (y_test[i] == 1)
        else:
            num_correct += (y_test[i] == 0)

    accuracy = num_correct / total
    return accuracy

def main():

    # Load the training set
    df_train = load("stress-train.csv")

    # Compute model parameters
    thetas = gradient_ascent(df_train, "stress_level")

    # Load the test set
    df_test = load("stress-test.csv")


    print(f"Training accuracy: {compute_accuracy(df_train, thetas)}")
    print(f"Test accuracy: {compute_accuracy(df_test, thetas)}")

    log_likelihood = 0
    for index, row in df_train.iterrows():
        xs = [1] + row[:-1].tolist()
        log_likelihood += (row["stress_level"] * math.log(sigmoid(transpose(thetas, xs)))) + ((1-row["stress_level"]) * math.log(1 - (sigmoid(transpose(thetas, xs)))))
    print(log_likelihood)

    print(thetas)
    sorted_thetas = sorted(thetas)
    print(sorted_thetas)

if __name__ == "__main__":
    main()
