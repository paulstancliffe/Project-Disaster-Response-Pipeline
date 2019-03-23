import sys
import pickle
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re

import warnings
warnings.filterwarnings(action='ignore')

import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    '''
    INPUTS:
    database_filpath - location of database to use to train model
    OUTPUTS:
    X - list of messages to be used to train the model
    Y - labels to be used to train the model
    classes - a list of label category names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageToCategories',con=engine)
    X = df['message'].values 
    Y = df.iloc[:,4:]
    classes = Y.columns
    return X, Y, classes 


def tokenize(text):
    '''
    Function that normalizes, tokenizers and lemmatizers messages.
    INPUTS:
    text - message as a string
    OUTPUTS:
    clean_tokens - message split into individual lemmatised words each as a string in a list
    '''
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Function builds model for training
    OUTPUT:
    cv : returns best model 
    '''
    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced',max_iter=20000)))
                        ])
    
    # Parameters to fine tune model
    parameters = {'vect__ngram_range':[(1,1),(1,2),(1,3)],
              'clf__estimator__C':[0.75,0.9,1.0]
              }
    
    # Train model and GridSearch for best parameters
    cv = GridSearchCV(pipeline,param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUTS:
    model - trained model
    X_test - list of messages to be used to test the trained model
    Y_test - labels to be used to test the trained model
    category_names - a list of label names
    '''
    # get predictions 
    Y_preds = model.predict(X_test)
    
    # print classification report and overall accuracy
    print(classification_report(Y_preds, Y_test.values, target_names=category_names))
    overall_accuracy = (Y_preds == Y_test).mean().mean()
    print("\t\t---------------")
    print('Average Overall Accuracy : {0:.2f}% \n'.format(overall_accuracy*100))


def save_model(model, model_filepath):
    '''
    INPUT:
    model - trained model
    model_filepath - filepath to save model
    OUPUT:
    a saved file in python pickle format
    '''
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=19,shuffle=True)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
