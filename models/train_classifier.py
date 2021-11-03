# import libraries

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import time
from datetime import datetime
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from sqlalchemy import create_engine
import sqlite3
from sqlalchemy import inspect

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from adicionalFeatures import TextLengthExtractor
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
import joblib

def load_data(database_file_path):
    
    """
    Load data from sqlite database then covert to X(traing data) and  Y(target labels)
    
    INPUT:
    database_file_path: data base file path
    
    OUTPUT:
    seriers: X, messages
    seriers: Y, labels
    labels: label names y
    """
    
    engine = create_engine('sqlite:///'+database_file_path)
    inspector = inspect(engine)

    # Get table information

    # load data from database
    conn = sqlite3.connect(database_file_path)
    df = pd.read_sql('SELECT * FROM '+inspector.get_table_names()[0], con = conn)
    X = df['message'].values
    y = df[df.columns[4:]].values
    
    return X,y,list(df.columns[4:])

def tokenize(text):
    
    """
    tokenize and cleand the text
    
    INPUT:
    text: string
    
    OUTPUT:
    clean_tokens: list of clean tokens
    """
    
    # text transformation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = text.strip()
    
    #stop ords
    stop_words = stopwords.words("english")
    
    #tokenize and lematize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #detele stop words
    clean_tokens =  [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Add the length of the text message as a feature to dataset
    
    The assumption is people who is in urgent disaster condition will prefer to use less words to express
    '''
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).applymap(len)

def build_model():
    """
    Builds a classification pipeline use random grid search
    
    INPUT:
    none
    
    OUTPUT:
    model: The best model of the random grid
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),

            ('txt_len', TextLengthExtractor()) #new feature
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    parameters = {
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__max_depth': [10,50, None],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [100,1000]
    }
    
    # takes a long time running.
    # model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter = 100,
    #                            cv = 5, verbose=2, random_state=42, n_jobs = -1)
    
    model = pipeline
    
    return model
    
def evaluate_model(model, X_test, y_test, labels):
    """
    Evaluate model on test set and predict
    
    INPUT:
    model: trained model
    X_test: pandas.DataFrame for predict 
    y_test: pandas.DataFrame labeled test set
    labels: list for category names
    
    OUTPUT:
    results pandas.DataFrame with the results.
    """
    
    y_pred = model.predict(X_test) # predict
    #create a dataFrame for save the results
    results = pd.DataFrame(columns=['column','precision','recall','f1-score'],dtype=float)

    # save the results for each category name
    for i, col in enumerate(labels):    
        results.loc[i,'column']=col
        results.loc[i,['precision','recall','f1-score']] = list(precision_recall_fscore_support(
            y_test[:,i], y_pred[:,i], average = 'weighted'))[0:3]

    # find the average of the result and save in the last row
    results.loc[i+1,['precision','recall','f1-score']] = results.mean(numeric_only=True)
    results.loc[i+1,'column']='average'
    
    return results

def save_model(model, model_name):
    """
    save the model in pkl format
    
    INPUT:
    model: a sklearn model.
    model_name: the model name for save model.
    
    OUTPUT:
    none.
    """
    joblib.dump(model, model_name+'.pkl', compress = 1)

def words_without_stopwords(text_list):
    
    """
    create a long text with all words without stop words.
    
    INPUT:
    text_list: list of texts
    
    OUTPUT:
    all_text: list of all words
    """
    
    # text transformation

    stop_words = stopwords.words("english")
    all_words=[]

    for text in tqdm(text_list):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]"," ",text)
        text = text.strip()
    
        tokens = word_tokenize(text)

        clean_tokens = [word for word in tokens if word not in stop_words]
        for word in clean_tokens:
            all_words.append(word)

    return all_words
    
def main():
    
    print('Loading data...')
    X, y, labels = load_data('../data/DisasterResponse.db')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Building model...')
    model = build_model()
	
    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    print('------ Results of the model training -------')
    print(evaluate_model(model, X_test, y_test, labels))
    print('------------------------------------------------')

    print('Saving model...')
    save_model(model, 'classifier')

    print('Calculing the most common words...')
    all_words = Counter(words_without_stopwords(list(X)))
    count_words = pd.DataFrame(list(dict(all_words.most_common(100)).keys()),columns=['words'])
    count_words['frequency'] = list(dict(all_words.most_common(100)).values())
    print('Saving the most common words...')
    count_words.to_csv('count_words.csv',index=False)

    print('Done!')

if __name__ == '__main__':
    main()