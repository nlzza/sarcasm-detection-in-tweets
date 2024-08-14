import numpy as np 
import pandas as pd 
from sklearn.svm import LinearSVC
import seaborn as sns
import spacy
import matplotlib.pyplot as plt
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
global i

nlp = spacy.load('en_core_web_sm')    
data = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
data.head()
data.isnull().sum()

#print("Features: ", data.headline)
#print("Labels: ", data.is_sarcastic)

def traverse_headline(text):    
    doc=nlp(text)
    print(f'Headline : {doc}')
    print(f'\nTotal number of tokens : {len(doc)} \n')

    for token in doc:
        print(token.text,end=' | ')

    for token in doc:
        print(f'{token.text:{12}}{token.pos_:{10}}{token.dep_:{12}}{str(spacy.explain(token.dep_))}')
    
    print(f'\nTotal number of Sentences : {len(list(doc.sents))}')
    for sent in doc.sents:
        print(sent)
        
    if len(doc.ents)>0:
        print(f'\nTotal number of Entity : {len(doc.ents)}\n')    
        for ent in doc.ents:
             print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
        displacy.render(doc,style='ent',jupyter=True)
    
    displacy.render(doc,style='dep',jupyter=True,options={'distance': 80}) 
    
def get_ents(text):
    doc=nlp(text)
    return len(doc.ents)

def get_tokens(text):
    doc=nlp(text)
    return len(doc)

def get_sents(text):
    doc=nlp(text)
    return len(list(doc.sents))

#traverse_headline(data['headline'][0])
#traverse_headline(data['headline'][1])

data_sample = data.sample(frac=.30,random_state=1)
data_sample.head()

data_sample['ents_num'] = data_sample['headline'].apply(get_ents)
data_sample['tokens_num'] = data_sample['headline'].apply(get_tokens)
data_sample['sents_num'] = data_sample['headline'].apply(get_sents)
data_sample.head()

fig,(ax,ax1,ax2)=plt.subplots(nrows=3,ncols=1,figsize=(15,15))
sns.countplot(x='ents_num',data=data_sample,hue='is_sarcastic',ax=ax,palette='spring')
sns.countplot(x='tokens_num',data=data_sample,hue='is_sarcastic',ax=ax1,palette='winter')
sns.countplot(x='sents_num',data=data_sample,hue='is_sarcastic',ax=ax2,palette='cool')

#building classification model

data_sample.drop(['article_link','ents_num','tokens_num','sents_num'],axis=1,inplace=True)
data.drop(['article_link'],axis=1,inplace=True)

blanks = []
for i,he,is_sa in data_sample.itertuples():
    if type(he) == str:
        if he.isspace():
            blanks.append(i)
#print(len(blanks), 'blanks: ', blanks)

#train_test_split
X = data_sample['headline']
y = data_sample['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Building a pipeline
classifier_svc = Pipeline([('tfidf',TfidfVectorizer()),
                     ('clf',LinearSVC())])

classifier_svc.fit(X_train,y_train)
classifier_svc.fit(X_test,y_test)

y_pred = classifier_svc.predict(X_test)
yt_pred = classifier_svc.predict(X_train)

#print("Test predict :", y_pred)
print("Train predict :", yt_pred)

#print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')
#print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')

my_index=0

user_tweet = input("user tweet: ") 
#print(user_tweet)
my_index = y_pred.shape[0]-1
X_test[my_index] = user_tweet

classifier_svc.fit(X_train,y_train)
y_pred = classifier_svc.predict(X_test)

print("result2: ", y_pred[my_index])
traverse_headline(y_pred[my_index])

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')
print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')


from tkinter import *
  
root = Tk()
root.geometry("500x300")
root.title('Sarcams Detector For Twitter')
t_var=StringVar() 
w = Label(root, text='Sarcasm Detector')
w.pack()

e = Entry(root, textvariable = t_var,font=('calibre',10,'normal'))
e.pack(side = TOP, ipadx = 150, ipady = 100,) 
e.focus_set()
frame = Frame(root) 
frame.pack() 
bottomframe = Frame(root) 
bottomframe.pack(anchor="center") 

blackbutton = Button(bottomframe, text ='Check', fg ='black') 
blackbutton.pack() 
root.mainloop()