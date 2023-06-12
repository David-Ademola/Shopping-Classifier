import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    x_train, x_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.4)

    # Train model and make predictions
    model = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    predictions = model.predict(x_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    # Loads data from a CSV file and preprocesses it for machine learning.

    ### Args:
        `filename` (str): The name of the CSV file to load.

    ### Returns:
        - tuple: A tuple containing two lists:
            - `evidence` (list): A list of lists representing the preprocessed features.
            - `labels` (list): A list of labels corresponding to each row of evidence.

    ### Raises:
        `FileNotFoundError`: If the specified file does not exist.

    ### Example:
        `evidence, labels = load_data('data.csv')`
    """
        
    MONTHS = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    INT_LABELS = {'Administrative', 'Informational', 'ProductRelated', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'}
    FLOAT_LABELS = {'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'}
    evidence, labels = [], []
    
    try:
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Convert 'Month', 'VisitorType', 'Weekend' and 'Revenue' to numeric values
                row['Month'] = MONTHS[row['Month']]
                row['VisitorType'] = int(row['VisitorType'] == 'Returning_Visitor')
                row['Weekend'] = int(row['Weekend'] == 'TRUE')
                row['Revenue'] = int(row['Revenue'] == 'TRUE')

                # Convert selected labels to integer or float
                for label in INT_LABELS: row[label] = int(row[label])
                for label in FLOAT_LABELS: row[label] = float(row[label])

                # Append row to evidence and labels lists
                evidence.append(list(row.values())[:17])
                labels.append(row['Revenue'])

        return evidence, labels
    
    except FileNotFoundError:
        raise FileNotFoundError(f'The specified file "{filename}" does not exist.')


def evaluate(labels, predictions):
    """
    # Calculates sensitivity and specificity based on the provided labels and predictions.

    ### Args:
        - `labels` (list): A list of labels representing the true class values (0 or 1) for each sample.
        - `predictions` (list): A list of predictions representing the predicted class values (0 or 1) for each sample.

    ### Returns:
        - `sensitivity` (float): The sensitivity, also known as true positive rate or recall.
        - `specificity` (float): The specificity, also known as true negative rate.

    ### Raises:
        - `ValueError`: If the lengths of `labels` and `predictions` are not equal.

    ### Example:
        - `labels` = [0, 1, 1, 0]
        - `predictions` = [0, 0, 1, 1]
        - `sensitivity, specificity = evaluate(labels, predictions)`
        #### Returns:
        - sensitivity = 0.5
        - specificity = 0.5

    The function calculates sensitivity and specificity to evaluate the performance of a binary classification model.
    Sensitivity measures the proportion of correctly predicted positive samples out of all actual positive samples.
    Specificity measures the proportion of correctly predicted negative samples out of all actual negative samples.
    The function counts the number of actual positive and negative labels and then compares the labels and predictions
    to calculate the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) counts.
    Sensitivity is calculated as TP / (TP + FN), and specificity is calculated as TN / (TN + FP).
    """
    if len(labels) != len(predictions):
        raise ValueError("The lengths of 'labels' and 'predictions' must be equal.")

    sensitivity, specificity, actual_positive_labels, actual_negative_labels = 0, 0, 0, 0

    # Count the number of actual positive and negative labels
    for label in labels:
        if label == 1:
            actual_positive_labels += 1
        else:
            actual_negative_labels += 1

    # Calculate sensitivity and specificity based on labels and predictions
    for label, prediction in zip(labels, predictions):
        if label == prediction == 1: sensitivity += 1
        if label == prediction == 0: specificity += 1
    
     # Divide sensitivity and specificity by the respective counts to obtain proportions
    sensitivity /= actual_positive_labels
    specificity /= actual_negative_labels

    return sensitivity, specificity


if __name__ == "__main__":
    main()