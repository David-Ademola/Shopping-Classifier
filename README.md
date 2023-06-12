# Shopping Classifier

The Shopping Classifier is a Python script that uses the k-nearest neighbors algorithm from the scikit-learn library to classify shopping data. It loads data from a CSV file, preprocesses it, splits it into training and testing sets, trains a KNN classifier, and evaluates its performance.

## Prerequisites

- Python 3.6 or above
- Scikit-learn library

## Usage

1. Clone the repository or download the script file (`shopping.py`).
2. Install the required dependencies by running the following command:
- `pip install -U scikit-learn`
3. Prepare your data in a CSV file, making sure it follows the required format. The file should contain both the features (evidence) and the corresponding labels.

**Note:** The script expects the last column in the CSV file to be the label column.

4. Run the script using the following command:
- `python shopping.py <data_file.csv>`

Replace `<data_file.csv>` with the path to your CSV file.

## Functionality

The script performs the following steps:

1. Checks the command-line arguments to ensure the correct usage.
2. Loads the data from the provided CSV file and preprocesses it for machine learning.
3. Splits the data into training and testing sets using a 60:40 ratio.
4. Trains a KNN classifier with one neighbor using the training set.
5. Makes predictions on the testing set.
6. Evaluates the performance of the classifier by calculating sensitivity (true positive rate) and specificity (true negative rate).
7. Prints the results, including the number of correct and incorrect predictions, true positive rate, and true negative rate.

## Customization

If you want to modify the behavior of the script, you can make the following changes:

- Change the `n_neighbors` parameter in the `KNeighborsClassifier` constructor to adjust the number of neighbors used for classification.
- Modify the `MONTHS`, `INT_LABELS`, and `FLOAT_LABELS` dictionaries in the `load_data` function to customize the mapping of categorical and numerical labels in the CSV file.

## License

This script is provided under the [MIT License](LICENSE).

## Issues

If you encounter any issues or have suggestions for improvements, please [create an issue](https://github.com/David-Ademola/Shopping-Classifier/issues) on the repository.

## Acknowledgments

- The script uses the Scikit-learn library for the K-nearest neighbors classifier.
- Data set provided by [Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007%2Fs00521-018-3523-0)


# 📖 Understanding 

When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn’t planning to complete the purchase. How could a website determine a user’s purchasing intent? That’s where machine learning will come in.

I built a nearest-neighbor classifier to solve this problem. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — my classifier predicted whether or not the user will make a purchase. My classifier wasn’t be perfectly accurate — perfectly modeling human behavior is a task well beyond the scope of this project — but it was better than guessing randomly. To train my classifier, I downloaded some data from a shopping website from about 12,000 users sessions.

How do I measure the accuracy of a system like this? If I have a testing data set, I could run the classifier on the data, and compute what proportion of the time the classifier correctly classify the user’s intent. This would give me a single accuracy percentage. But that number might be a little misleading. Imagine, for example, if about 15% of all users end up going through with a purchase. A classifier that always predicted that the user would not go through with a purchase, then, it would measure as being 85% accurate: the only users it classifies incorrectly are the 15% of users who do go through with a purchase. And while 85% accuracy sounds pretty good, that doesn’t seem like a very useful classifier.

Instead, I measured two values: sensitivity (also known as the “true positive rate”) and specificity (also known as the “true negative rate”). Sensitivity refers to the proportion of positive examples that were correctly identified: in other words, the proportion of users who did go through with a purchase who were correctly identified. Specificity refers to the proportion of negative examples that were correctly identified: in this case, the proportion of users who did not go through with a purchase who were correctly identified. So our “always guess no” classifier from before would have perfect specificity (1.0) but no sensitivity (0.0). My goal was to build a classifier that performs reasonably on both metrics.