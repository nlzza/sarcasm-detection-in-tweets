import pandas as pd
import spacy
import nltk
import re
from senticnet.senticnet import SenticNet
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from funcs import *

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Load SentiStrength and SenticNet
sentic_net = SenticNet()

# Load data
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
data = data.sample(frac = .30, random_state = 1)

# Preprocess the data
data['headline'] = data['headline'].apply(lambda x: re.sub(r'\W', ' ', x))  # Remove non-word characters
X = data['headline']
y = data['is_sarcastic']

feature_vectors = []
labels = []

for tweet in X:
    # Module 1: Concept Level and Common-Sense Knowledge
    semantic_analysis = analyze_concepts(tweet)

    # Module 2: Contradiction in the Sentiment Score
    sentiment_scores = [get_sentiment_score(word) for word in nltk.word_tokenize(tweet)]

    # Module 4: Creation of Feature Vector
    feature_vector = create_feature_vector(tweet)

    # Append the module results and feature vector to the lists
    combined_features = feature_vector + semantic_analysis + sentiment_scores
    feature_vectors.append(combined_features)
    labels.append(y.loc[X == tweet].values[0])

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size = 0.33, random_state = 42)
X_train = [' '.join(map(str, x)) for x in X_train]
X_test = [' '.join(map(str, x)) for x in X_test]
# Build the pipeline
classifier_svc = Pipeline([('tfidf', TfidfVectorizer(ngram_range = (1, 3))),
                           ('clf', LinearSVC())])

# Fit the model on training data
classifier_svc.fit(X_train, y_train)

# Predict on test data
y_pred = classifier_svc.predict(X_test)
yt_pred = classifier_svc.predict(X_train)

# Print the predictions and accuracy scores
print("\nTest predict:", y_pred)
print("Train predict:", yt_pred)
print("\nTest Set Accuracy Score:", accuracy_score(y_test, y_pred))
print("Train Set Accuracy Score:", accuracy_score(y_train, yt_pred))

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

# Generate the classification report
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report:\n")
print(cr)

# Create UI using Tkinter
root = Tk()
root.geometry("500x300")
root.title('Sarcasm Detector For Twitter')

# Function to perform the check
def perform_check():
    user_tweet = t_var.get()
    X_test = [user_tweet]
    y_pred = classifier_svc.predict(X_test)
    print(y_pred)
    output = "Sarcastic" if y_pred else "Not Sarcastic"
    result_label.config(text = f"Result: {output}")  # Update the result_label with the predicted result

# UI components
w = Label(root, text = 'Sarcasm Detector')
w.pack()
t_var = StringVar()
e = Entry(root, textvariable = t_var, font = ('calibre', 10, 'normal'))
e.pack(side = TOP, ipadx = 150, ipady = 100)
e.focus_set()
frame = Frame(root)
frame.pack()
bottomframe = Frame(root)
bottomframe.pack(anchor = "center")
blackbutton = Button(bottomframe, text = 'Check', fg = 'black', command = perform_check)
blackbutton.pack()
result_label = Label(root, text = "")
result_label.pack()

root.mainloop()
