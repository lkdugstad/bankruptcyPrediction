import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# install mlserver with pip
# start server with mlserver start .


def request_to_model():
    df = pd.read_csv('data/cleandata.csv')
    X = df.drop('is_bankrupt_after_years', axis=1)
    y = df['is_bankrupt_after_years']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0002, stratify=y, random_state=42)
    inference_request = {'inputs': [{'name': 'predict', 'shape': X_test.shape,
                                     'datatype': 'FP64', 'data': X_test.values.tolist()}]}
    endpoint = 'http://localhost:8080/v2/models/bankrupt-sklearn/versions/v1/infer'
    response = requests.post(endpoint, json=inference_request)
    print(response.text)
    print(y_test.values)


if __name__ == '__main__':
    request_to_model()