import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string


def decision_tree():
    dataset = load_dataset()
    dataset['Category'] = change_category(dataset)
    dataset['Message'].apply(clean_msg)
    X_train, X_test, Y_train, Y_test = train_model(dataset)
    X_train_vct, X_test_vct = tfidf_feature(X_train, X_test)
    DT = make_dt_model(X_train_vct, Y_train)
    return DT, X_train


def make_dt_model(X_train_vct, Y_train):
    DT = DecisionTreeClassifier(
        max_depth=5, criterion='entropy', random_state=42)
    DT.fit(X_train_vct, Y_train)
    return DT


def tfidf_feature(X_train, X_test):
    vct = TfidfVectorizer()
    X_train_vct = vct.fit_transform(X_train)
    X_test_vct = vct.transform(X_test)
    return X_train_vct, X_test_vct


def train_model(dataset):
    X = dataset['Message']
    Y = dataset['Category']
    return train_test_split(X, Y, test_size=0.3)


def clean_msg(Message):
    Message = Message.lower()
    Message = re.sub('\[.*?\]', '', Message)
    Message = re.sub('https?://\S+|www\.\S+', '', Message)
    Message = re.sub('<.*?>+', '', Message)
    Message = re.sub('[%s]' % re.escape(string.punctuation), '', Message)
    Message = re.sub('\n', '', Message)
    Message = re.sub('\w*\d\w*', '', Message)
    return Message


def load_dataset():
    return pd.read_csv('./dataset/test-dataset.csv')


def change_category(dataset):
    return dataset['Category'].map({'ham': 0, 'spam': 1})
