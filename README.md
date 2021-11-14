# Auto-Machine-learning-Clustering
Auto-Clustering
 program structure that make it possible to automatically run 
different combinations of: 1) Various data scaling methods and encoding methods
                            2) Various values of the model parameters for each model
                            3) Various values for the hyperparameters
                            4) Various subsets of the features of the dataset
                            
Considered models includes: K-means ,EM(GMM), CLARANS ,DBSCAN, SpectralClustering and clarans

scalers: list of scalers
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            
encoders: list of encoders
        None: [OrdinalEncoder(), OneHotEncoder(),SVC(),]
