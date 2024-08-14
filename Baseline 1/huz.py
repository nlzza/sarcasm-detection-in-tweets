import pandas as pd
from sklearn.svm import LinearSVC
import seaborn as sns
import spacy
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Load data
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)


# Function to traverse headline and display information
def traverse_headline(text):
    text = str(text)
    doc = nlp(text)
    print(f'Headline : {doc}')
    print(f'\nTotal number of tokens : {len(doc)} \n')

    for token in doc:
        print(token.text, end=' | ')

    for token in doc:
        print(f'{token.text:{12}}{token.pos_:{10}}{token.dep_:{12}}{str(spacy.explain(token.dep_))}')

    print(f'\nTotal number of Sentences : {len(list(doc.sents))}')
    for sent in doc.sents:
        print(sent)

    if len(doc.ents) > 0:
        print(f'\nTotal number of Entity : {len(doc.ents)}\n')
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))

    output_path = "dependency_visualization.png"
    displacy.render(doc, style='dep', options={'distance': 80, 'compact': True, 'color': 'black'}, jupyter=False)
    plt.savefig(output_path)


# Function to get number of entities
def get_ents(text):
    doc = nlp(text)
    return len(doc.ents)


# Function to get number of tokens
def get_tokens(text):
    doc = nlp(text)
    return len(doc)


# Function to get number of sentences
def get_sents(text):
    doc = nlp(text)
    return len(list(doc.sents))


traverse_headline(data['headline'][0])
# Sample data
data_sample = data.sample(frac=.30, random_state=1)

# Add columns for number of entities, tokens, and sentences
data_sample['ents_num'] = data_sample['headline'].apply(get_ents)
data_sample['tokens_num'] = data_sample['headline'].apply(get_tokens)
data_sample['sents_num'] = data_sample['headline'].apply(get_sents)

# Create visualization of entity, token, and sentence counts
fig, (ax, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
sns.countplot(x='ents_num', data=data_sample, hue='is_sarcastic', ax=ax, palette='spring')
sns.countplot(x='tokens_num', data=data_sample, hue='is_sarcastic', ax=ax1, palette='winter')
sns.countplot(x='sents_num', data=data_sample, hue='is_sarcastic', ax=ax2, palette='cool')

# Building classification model
data_sample.drop(['article_link', 'ents_num', 'tokens_num', 'sents_num'], axis=1, inplace=True)
data.drop(['article_link'], axis=1, inplace=True)

# Split the data into train and test sets
X = data_sample['headline']
y = data_sample['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Build the pipeline
classifier_svc = Pipeline([('tfidf', TfidfVectorizer()),
                           ('clf', LinearSVC())])

# Fit the model on training data
classifier_svc.fit(X_train, y_train)

# Predict on test data
y_pred = classifier_svc.predict(X_test)
yt_pred = classifier_svc.predict(X_train)

# Print the predictions
print("Test predict :", y_pred)
print("Train predict :", yt_pred)

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test, y_pred)}\n')
print(f'Train Set Accuracy Score :\n {accuracy_score(y_train, yt_pred)}\n')

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test, y_pred)}\n')
print(f'Train Set Accuracy Score :\n {accuracy_score(y_train, yt_pred)}\n')

# Create UI using Tkinter
root = Tk()
root.geometry("500x300")
root.title('Sarcasm Detector For Twitter')


# Function to perform the check
# Function to perform the check
def perform_check():
    user_tweet = t_var.get()
    X_test = [user_tweet]
    y_pred = classifier_svc.predict(X_test)
    result_label.config(text=f"Result: {y_pred}")


# UI components
w = Label(root, text='Sarcasm Detector')
w.pack()
t_var = StringVar()
e = Entry(root, textvariable=t_var, font=('calibre', 10, 'normal'))
e.pack(side=TOP, ipadx=150, ipady=100)
e.focus_set()
frame = Frame(root)
frame.pack()
bottomframe = Frame(root)
bottomframe.pack(anchor="center")
blackbutton = Button(bottomframe, text='Check', fg='black', command=perform_check)
blackbutton.pack()
result_label = Label(root, text="")
result_label.pack()

root.mainloop()
