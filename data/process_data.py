# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load data and merge in a data Frame

    INPUT:
    messages.csv
    categories.csv
    
    OUTPUT:
    df : merged data frame of the messages and categories csv
    """
    
    # load messages
    messages = pd.read_csv(messages_filepath)
    # load categories
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on='id')
    # split categories into separate category columns
    categories.set_index('id',inplace=True)
    categories = categories.categories.str.split(";",expand=True)
    
    # extract a list of the new column names for categories
    row = categories.index.values[0]
    category_colnames = categories.loc[row,:].apply(lambda x:x.split('-')[0]).values
    
    #change the name of the columns of data frame categories
    categories.columns = category_colnames
    
    #convert category values to numbers (0 or 1)
    
    for column in categories:
        # extract the number
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories.reset_index(inplace=True) # important for merge with df dataframe
    
    # droping the categories column from the df dataframe
    df.drop('categories', axis=1,inplace=True,errors='ignore')
    # replace categories column in df with new category columns
    df = df.merge(categories,on='id')
    df = df[df['related']<=1]
    df.reset_index(drop=True,inplace=True)
    
    return df

def drop_duplicates(df):
    """
    drop duplicates of the data frame

    INPUT:
    df : data frame with duplicates
    
    OUTPUT:
    df : data frame without duplicates
    """

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_name):
    """
    save date frame in a database

    INPUT:
    df : data frame with duplicates
    database_name : name of the data base
    
    """

    # save the clean dataset into an sqlite database
    db_path= "sqlite:///"+database_name+".db"
    engine = create_engine(db_path)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')
    
def main():
    print('Loading data...')
    df = load_data('messages.csv','categories.csv')

    print('Removing duplicates...')
    df = drop_duplicates(df)

    print('Saving data in database....')
    save_data(df, 'DisasterResponse')

    print('Done!')

if __name__ == '__main__':
    main()