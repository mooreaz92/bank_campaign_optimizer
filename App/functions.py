import pandas as pd
import numpy as np

################################ CLEANING FUNCTIONS ################################

### Making functions that handle the categorical features in the data set

def cast_as_columns(df):
    columns_to_categorize = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
    for column in columns_to_categorize:
        df[column] = df[column].astype('category')
    return df

def ordinal_encode_education(df):
    df['education'] = df['education'].cat.reorder_categories(['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
                                                              'high.school', 'professional.course', 'university.degree'])
    df['education'] = df['education'].cat.codes
    return df

def encode_target(df):
    df['y'] = df['y'].cat.codes
    return df

### Making a function that drops the 'duration' and 'default' features from the dataframe, since they are not useful for the model and contain too many missing values

def drop_features(df):
    df = df.drop(['duration', 'default'], axis=1)
    return df

### Making a funtion that drops the consumer price index and the number of employees features, since they are highly correlated with each other and not correlated with the target variable

def drop_features_corr(df):
    df = df.drop(['cons.price.idx', 'nr.employed'], axis=1)
    return df


### Making a function that sets the 999 value in the 'pdays' feature to 0, since it means the client was not contacted before

def set_pdays_to_zero(df):
    df['pdays'] = df['pdays'].replace(999, 0)
    return df

### Writing a function that combines the above functions

def clean_data(df):
    df = cast_as_columns(df)
    df = ordinal_encode_education(df)
    df = drop_features(df)
    df = set_pdays_to_zero(df)
    df = drop_features_corr(df)
    df = encode_target(df)
    return df

################################ PREPROCESS FUNCTIONS ################################

### Writing a function that performs a one hot encodes the categorical features of df and drops the original categorical features

def one_hot_encode(df):
    for column in df.columns:
        if df[column].dtype == 'category':
            df = pd.get_dummies(df, columns=[column], prefix=column)
    return df

### Writing a function that performs a standard scaler on the numerical features of df

from sklearn.preprocessing import StandardScaler

def standard_scale(df):
    for column in df.columns:
        if df[column].dtype != 'category':
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    [column] = scaler.transfor[column].values.reshape(-1, 1)
    return df

### Writing a function that combines the above functions

def preprocess_data(df, y_train, y_test):
    df = one_hot_encode(df)
    df = standard_scale(df)
    return df, y_train, y_test
