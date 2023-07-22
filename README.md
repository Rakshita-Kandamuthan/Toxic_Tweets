# Toxic Tweets

This repository contains Python code for performing text analysis and building machine learning models to classify toxic tweets. The code uses various natural language processing (NLP) techniques and popular classifiers from scikit-learn to accomplish this task.

## Prerequisites
Before running the code, ensure that you have the following installed:

   - Python 3.x
   - **Required Python libraries**: scikit-learn, numpy, matplotlib, pandas, nltk


   You can install the required libraries using pip:
   ```
   !pip install scikit-learn 
   !pip install numpy 
   !pip install matplotlib
   !pip install pandas
   !pip install nltk 
   ```
Additionally, make sure to download the necessary NLTK resources:
  ```
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')

   ```

## Dataset

The code reads data from a CSV file containing tweets and their associated toxicity labels. The dataset should be formatted with two columns: 'tweet' (containing the text of tweets) and 'Toxicity' (binary labels indicating whether a tweet is toxic or not).

The path to the dataset CSV file should be provided in the code before running.

## Text Cleaning

The tweet text is preprocessed through the following steps:

- Convert text to lowercase.
- Tokenize the text to obtain individual words (alphanumeric characters only).
- Remove English stopwords.
- Lemmatize words to convert them to their base form.

## Bag of Words Representation

The cleaned tweet text is converted into a Bag of Words (BoW) representation using the CountVectorizer from scikit-learn.

## TF-IDF Representation

The cleaned tweet text is also converted into a TF-IDF (Term Frequency-Inverse Document Frequency) representation using the TfidfVectorizer from scikit-learn.


## Classification Models

The code builds and evaluates the performance of the following classification models on both BoW and TF-IDF representations:

   - Decision Tree Classifier
   - Random Forest Classifier
   - Naive Bayes Classifier (Gaussian Naive Bayes)
   - K-Nearest Neighbors (KNN) Classifier
   - Support Vector Machine (SVM) Classifier 


## Running the Code

1. Set the path to the dataset CSV file in the code:
   ```
   data = pd.read_csv(r"path/to/dataset.csv")

   ```

2. Ensure the required Python libraries are installed as mentioned in the Prerequisites section.

3. Run the code in a Python environment.

The code will train each classifier on the given dataset and print the following evaluation metrics for each model:

   - Precision
   - Recall
   - F1 - Score
   - Confusion Matrix
   - ROC-AUC Score 

Additionally, the code will plot the ROC curves for each classifier to visualize their performance.

### Note

Please note that the performance of the classifiers may vary depending on the dataset and the quality of the data. It is recommended to experiment with different classifiers and preprocessing techniques to achieve the best results for a specific task.

If you use this code or find it helpful, consider giving credits and references to this repository.

Feel free to modify and adapt the code as per your specific requirements and datasets.

Happy coding!



