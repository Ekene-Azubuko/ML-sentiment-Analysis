from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, request
import pandas as pd
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

app = Flask(__name__)

df = getDF('Grocery_and_Gourmet_Food.json.gz')

df['reviewText'].fillna('', inplace=True)
X = df['reviewText']  # Input features
y = df['overall'].apply(lambda x: 1 if x > 3 else 0)  # 1 for positive, 0 for negative sentiment
  # Binary classification (positive or negative sentiment)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)  # You can adjust parameters
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Predictions
predictions = model.predict(X_test_vect)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

#test sentiment
while True:
    try:
        # Read review text from console input
        text = input("Write a review (type 'exit' to quit): ")
        if text.lower() == 'exit':
            break  # Exit the loop if 'exit' is typed

        # Predict sentiment for the input text
        text_vect = vectorizer.transform([text])  # Vectorize the input text
        prediction = model.predict(text_vect)  # Predict sentiment
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Print the predicted sentiment
        print(f"Sentiment: {sentiment}\n")
    except Exception as e:
        print(f"An error occurred: {e}")

