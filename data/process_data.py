import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from csv files

    Input:
    messages_filepath: filepath where csv file is stored
    categories_filepath: filepath where category names are stored

    Output:
    df: pandas DataFrame with messages and output classes for each category name
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left = messages, right = categories, how = 'inner', on='id')
    return df


def clean_data(df):
    """
    Clean message data and category column names

    Input:
    df: pandas DataFrame with messages and categories

    Output:
    df: cleaned pandas DataFrame
    """
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe

    df = pd.concat(objs= [df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save message dataframe into a sqlite database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    """
    ETL pipeline, load, clean and save message data into sqlite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
