import pandas as pd

def calculate_mean_bmi(df, age, sex):
    group = df[(df['age'] == age) & (df['gender'] == sex)]
    mean_bmi = group['bmi'].median()
    return mean_bmi


def clean_data(df):
    df.drop('id', axis=1, inplace=True)
    for idx, row in df.iterrows():
        if pd.isnull(row['bmi']):
            age = row['age']
            sex = row['gender']
            mean_bmi = calculate_mean_bmi(df, age, sex)
            df.at[idx, 'bmi'] = mean_bmi
    df.dropna(inplace=True)
    return df


def deal_with_categorical_features(df):
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
    categorical_columns = ['gender', 'work_type', 'Residence_type', 'smoking_status']
    data_new = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    y = data_new.pop('stroke')
    data_new['stroke'] = y
    return data_new


def cor_features(df):
    highly_correlated_features = set()
    correlation_matrix = df.corr().abs()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > 0.6:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)
    print("Highly correlated features:", highly_correlated_features)


def signif_features(df):
    significant_features = df.corr()['stroke'].abs().sort_values(ascending=False)
    significant_features = significant_features[significant_features > 0.1].index.tolist()
    for i in significant_features:
        print(i)
    return significant_features

