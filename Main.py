from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("Tweet_Final.csv")
# Load the model from disk
loaded_model = load('model.joblib')
# Load the model from disk
data['clean_tweet'] = data['clean_text'].apply(lambda x: x.lower())
data['clean_tweet'] = data['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

X_train, X_test, y_train, y_test = train_test_split(data['clean_tweet'], data['category'], test_size=0.2, random_state=42)


# Initialize the vectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit(X_train)

# Use the model to make predictions on new data
tweet = input("Enter the tweet: ")

# preprocess the input tweet
c_tweet=re.sub(r"@user","",tweet)
c_tweet=c_tweet.replace("[^a-zA-Z#]"," ")
words = c_tweet.split()
filtered_words = [word for word in words if len(word) >= 3]
filtered_string = ' '.join(filtered_words)
new_data = vectorizer.transform([filtered_string])
predictions = loaded_model.predict(new_data)
print(predictions)
