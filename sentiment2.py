import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv("Tweet_Final.csv")
data['clean_tweet'] = data['clean_text'].apply(lambda x: x.lower())
data['clean_tweet'] = data['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

X_train, X_test, y_train, y_test = train_test_split(data['clean_tweet'], data['category'], test_size=0.2, random_state=42)

# Initialize the vectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(X_train)

# Initialize the model
clf = LogisticRegression(max_iter=1000)

# Fit the model on the training data
clf.fit(X_train, y_train)
# Transform the test data
X_test = vectorizer.transform(X_test)

# Predict the labels on the test data
y_pred = clf.predict(X_test)


# Calculate the evaluation metrics
#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)

# Print the results
#print("Accuracy: {:.2f}".format(accuracy))
#print("Precision: {:.2f}".format(precision))
#print("Recall: {:.2f}".format(recall))
#print("F1-score: {:.2f}".format(f1))
from joblib import dump, load

# Save the model to disk
dump(clf, 'model.joblib')