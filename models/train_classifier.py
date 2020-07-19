import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load message data from sqlite database

    Input: database filepath

    Output:
    X: pandas Series with messages
    Y: pandas DataFrame with output class of each message
    category_names: name of each output class
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize message

    Input: text

    Output: list of cleaned tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build classifier
    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {
                  'clf__estimator__n_estimators': [10,20]
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=10)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print the classification report for each output class of the model

    Input:
    model: sklearn model
    X_test: pandas Series with messages in the test dataset
    Y_test: pandas DataFrame with output classes from test dataset
    category_names: list of output class name
    """

    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, index = Y_test.index, columns = Y_test.columns)
    for i in range(len(category_names)):
        print(70*'='+'\nFeature: ',category_names[i])
        print(classification_report(Y_test.iloc[:,i],Y_pred.iloc[:,i]))




def save_model(model, model_filepath):
    """
    Dump model into pickle file

    Input:
    model: sklearn model
    model_filepath: filepath where model should be stored
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Machine learning pipeline, build, train, evaluate and stores a model.

    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
