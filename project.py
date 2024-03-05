import pandas as pd
import numpy as np
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum, strip_tags, remove_stopwords, strip_multiple_whitespaces, strip_punctuation, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class my_model():

    def __init__(self):
        self.vectorizer1 = TfidfVectorizer(preprocessor=self.preprocess, lowercase=False)
        self.vectorizer2 = TfidfVectorizer(preprocessor=self.preprocess, lowercase=False)
        self.vectorizer3 = TfidfVectorizer(preprocessor=self.preprocess, lowercase=False)

    # ------------------
    # preprocessing function
    # ------------------        
    def preprocess(self, data):
        p = PorterStemmer()
        # strip unnecessary characters
        data = strip_punctuation(data)
        data = strip_multiple_whitespaces(data)
        data = strip_tags(data)
        data = strip_non_alphanum(data)

        # remove stopwords
        data = remove_stopwords(data)
        data = data.lower()
        data = p.stem(data)
        return data
    
    # ------------------
    # fit function
    # ------------------        
    def fit(self, X, y):
        # concat title and locarion
        X['title'] = X['title'] + ' ' + X['location']
        X = X.drop('location', axis=1)
        
        # Column transformer
        transformer = ColumnTransformer([('title', self.vectorizer1, 'title'), ('description', self.vectorizer2, 'description'), ('requirements', self.vectorizer3, 'requirements')], remainder='passthrough')

        # classifier pipeline and params grid
        pipe = Pipeline(steps=[('preprocessor', transformer), ('clf', SVC())])
        svc_param_grid = {
            'clf__kernel': ['rbf', 'linear'],
            "clf__class_weight": ["balanced", None],
            "clf__gamma": ["scale", "auto"]
        }

        # multithreaded grid search focusing on f1 scoring 
        self.grid_search = GridSearchCV(pipe, svc_param_grid, n_jobs=-1, cv=5, scoring='f1')
        self.grid_search.fit(X, y)
        return

    # ------------------
    # predict function
    # ------------------ 
    def predict(self, X):
        # concat title and locarion
        X['title'] = X['title'] + ' ' + X['location']
        X = X.drop('location', axis=1)

        #predict with selected model
        predictions = self.grid_search.predict(X)

        return predictions