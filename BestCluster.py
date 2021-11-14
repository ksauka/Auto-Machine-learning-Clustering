import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import time
import datetime

import warnings

warnings.filterwarnings(action='ignore')

def cv_silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    return score


# function for find the best accuracy given datas, params
def BestCluster(X, scalers, encoders, models, params_dict=None):

    """
 A program structure that make it possible to automatically run 
 different combinations of: 1) Various data scaling methods and encoding methods
                            2) Various values of the model parameters for each model
                            3) Various values for the hyperparameters
                            4) Various subsets of the features of the dataset
Considered models includes: K-means ,EM(GMM), CLARANS ,DBSCAN, SpectralClustering and clarans
scalers: list of scalers
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
encoders: list of encoders
        None: [OrdinalEncoder(), OneHotEncoder(),SVC(),]
    """
    # save the best accuracy each models
    global best_params
    global score
    score = -1
    best_accuracy = {}

    # find the best parameter by using grid search

    for scaler_key, scaler in scalers.items():
        print(f'--------scaler: {scaler_key}--------')
        X_scaled = scaler.fit_transform(X[get_numeric_col(X)])

        for encoder_key, encoder in encoders.items():
            print(f'------encoder: {encoder_key}------')
            X_encoded = X_scaled.copy()
            for str_col in get_string_col(X):
                X_encoded = encoder.fit_transform(X[str_col].to_numpy().reshape(-1, 1))
                X_encoded = np.concatenate((X_scaled, X_encoded.reshape(X_scaled.shape[0], -1)), axis=1)

            for model_key, model in models.items():
                print(f'----model: {model_key}----')
                start_time = time.time()  # for check running time

                if model_key == "clarans_model":
                    param_list = list(params_dict[model_key].keys())

                    clarans_params = params_dict[model_key]
                    for p1 in clarans_params[param_list[0]]:
                        for p2 in clarans_params[param_list[1]]:
                            for p3 in clarans_params[param_list[2]]:
                                temp_params = {param_list[0]: p1, param_list[1]: p2,
                                               param_list[2]: p3}

                                clarans_model = clarans(data=X_encoded, number_clusters=p1, numlocal=p2, maxneighbor=p3)
                                clarans_model.process()
                                cluster_result = clarans_model.get_clusters()

                                labels = []
                                for i in range(len(cluster_result)):
                                    for j in cluster_result[i]:
                                        labels.insert(j, i)

                                temp_score = silhouette_score(X_encoded, labels, metric='euclidean')

                                if temp_score > score:
                                    score = temp_score
                                    best_params = temp_params
                else:
                    # grid search
                    cv = [(slice(None), slice(None))]
                    grid = GridSearchCV(estimator=model, param_grid=params_dict[model_key], scoring=cv_silhouette_scorer, cv=cv)
                    grid.fit(X_encoded)
                    best_params = grid.best_params_
                    score = grid.best_score_

                print(f'params: {best_params}')

                # save the 3 highest accuracy and parameters each models
                save_len = 3
                save_len -= 1
                flag = False

                target_dict = {'score': score, 'model': model_key, 'scaler': scaler_key,
                               'encoder': encoder_key, 'param': best_params}
                # save accuracy if best_accuracy has less than save_len items
                if model_key not in best_accuracy.keys():
                    best_accuracy[model_key] = []
                if len(best_accuracy[model_key]) <= save_len:
                    best_accuracy[model_key].append(target_dict)
                    best_accuracy[model_key].sort(key=lambda x: x['score'], reverse=True)
                # insert accuracy for descending
                elif best_accuracy[model_key][-1]['score'] < score:
                    for i in range(1, save_len):
                        if best_accuracy[model_key][save_len - 1 - i]['score'] > score:
                            best_accuracy[model_key].insert(save_len - i, target_dict)
                            best_accuracy[model_key].pop()
                            flag = True
                            break
                    if flag is False:
                        best_accuracy[model_key].insert(0, target_dict)
                        best_accuracy[model_key].pop()

                print(f'score: {score}', end='')
                end_time = time.time()  # for check running time
                print(f'   running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    print(f'------train result------')
    displayResultDict(best_accuracy)

    return best_accuracy


def get_numeric_col(df):
    numeric_col_list = []

    for col_name in df.columns:
        if is_numeric_dtype(df[col_name].dtypes):
            numeric_col_list.append(col_name)

    return numeric_col_list


def get_string_col(df):
    string_col_list = []

    for col_name in df.columns:
        if is_string_dtype(df[col_name].dtypes):
            string_col_list.append(col_name)

    return string_col_list


def outlier_iqr(df):
    numeric_col_list = get_numeric_col(df)

    for col_name in numeric_col_list:
        q1, q3 = np.percentile(df[col_name], [25, 75])

        iqr = q3 - q1

        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        df = df[upper_bound > df[col_name]]
        df = df[df[col_name] > lower_bound]

    return df


def preprocessing():
    df = pd.read_csv(r'housing.csv')
    df_drop_NAN = df.dropna(axis=0)     # drop NAN
    df_drop_outlier = outlier_iqr(df_drop_NAN)
    df_drop_outlier.reset_index(drop=True, inplace=True)
    return df_drop_outlier


# function for set hyper parameters and run find_best
def train():
    X = preprocessing().iloc[850:1001, -2:]

    # 1. Scaler : Standard, MinMax, Robust

    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()

    # 2. Encoder : Label, One-Hot

    label_encoder = LabelEncoder()
    oneHot_encoder = OneHotEncoder(sparse=False)

    # 3. Model : Decision tree(entropy), Decision tree(Gini), Logistic regression, SVM

    kmeans = KMeans()
    dbscan = DBSCAN()
    em = GaussianMixture()
    spectral = SpectralClustering()

    # save scalers and models and hyper parameters in dictionary

    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}

    encoders = {"one-hot encoder": oneHot_encoder, "label encoder": label_encoder}

    models = {"kmeans": kmeans, "dbscan": dbscan,
              "em": em, 'spectral': spectral, 'clarans_model': None}


    params_dict = {"kmeans": {"n_clusters": range(2, 4), "tol": [1e-6, 1e-4, 1e-2, 1]},
                   "dbscan": {"eps": [0.2, 0.5, 0.8], "min_samples": [3, 5, 7, 9]},
                   "em": {"n_components": [1, 2, 3], "tol": [1e-5, 1e-3, 1e-1, 10]},
                   "spectral": {"n_clusters": range(2, 4), "gamma": [1, 2]},
                   "clarans_model": {"number_clusters": range(3, 4), "numlocal": [6, 8],
                                     "maxneighbor": [2, 4]}
                   }

    BestCluster(X, scalers, encoders, models, params_dict)


# function for display result_dict
def displayResultDict(result_dict):
    print(result_dict)
    for model_name, result_list in result_dict.items():
        print(model_name)
        for result in result_list:
            print(result)


if __name__ == "__main__":
    train()
