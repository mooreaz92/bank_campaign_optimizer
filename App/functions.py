import pandas as pd
import numpy as np

################################ CLEANING FUNCTIONS ################################

def cast_as_columns(df):
    columns_to_categorize = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    for column in columns_to_categorize:
        df[column] = df[column].astype('category')
    return df

def cast_as_columns_manual(df):
    columns_to_categorize = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    for column in columns_to_categorize:
        df[column] = df[column].astype('category')
    return df

### Making a function that ordinal encodes the 'education' feature to numerical values using a dictionary, starting with unknown as 0 and illiterate as 1, and so on

def ordinal_encode_education(df):
    df['education'] = df['education'].cat.reorder_categories(['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree'], ordered=True)
    df['education'] = df['education'].cat.codes
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

################################ PREPROCESS FUNCTIONS ################################

### Writing a function that performs a one hot encodes the categorical features of df and drops the original categorical features

def one_hot_encode(df):
    for column in df.columns:
        if df[column].dtype == 'category':
            df = pd.get_dummies(df, columns=[column], prefix=column)
    return df

### Writing a function that standard scales the numerical features of a dataframe

def standard_scale(df):
    from sklearn.preprocessing import StandardScaler
    for column in df.columns:
        if df[column].dtype != 'category':
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


