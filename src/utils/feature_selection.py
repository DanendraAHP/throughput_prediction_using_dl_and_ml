import pandas as pd
from sklearn.feature_selection import SelectFromModel, f_regression, RFE, SelectKBest
from sklearn.svm import SVR
from sklearn.decomposition import PCA

class Corr_selection():
    """
    Fungsi buat feature selection sebelum modeling
    input :
        df = data features yang mau dicari 
        target = target column names
        n_features = number of features to be included
        threshold = threshold to be included
    
    output :
        df = df yang sudah ditransformasi
    Jika memakai automatic maka menggunakan top 10 feature dengan abs(korelasi) terbesar
    """
    def __init__(self):
        pass
    
    def set_params(self, n_features, threshold):
        self.n_features = n_features
        self.threshold = threshold
        
    def fit(self, X, y):
        self.df = pd.concat([X,y], axis=1)
        self.corr = self.df.corr()[y.name]
        self.sorted_corr = self.corr.apply(abs).sort_values(ascending=False).drop(y.name)[:self.n_features]
        if self.threshold!=None:
            mask = self.sorted_corr>self.threshold
            self.sorted_corr = self.sorted_corr[mask]
            
    def get_feature_names_out(self,input_features):
        return self.sorted_corr.index
    
    def transform(self,X):
        return X[self.sorted_corr.index]

def feature_selection(X, y, how, **params):
    """
    make feature selection to reduce the columns
    Input :
        X : dataframe to be reduces
        y : dataframe that have target variable
        how : feature selection type
        n_features : number of features to be kept
    Output :
        dataframe transformed
        
    For 
    SelectKBest :
        Input :
            score_func : score function to used (
                f_classiff,  chi2, etc
            )
            k : top k features
    RFE :
        Input :
            estimator : model to see the feature importance
            n_features_to_select : features to included (if none then half the features will included)
            step : number of features to removed each step (if 0<step<1 then will become percentage)
    SelectFromModel :
        Input :
            estimator : model to see the feature importance
            threshold : feature importance to be included (can also be mean,median)
                        if none then use 1e-5
            max_features : maximum number of features used. To only select based on max_features, set threshold=-np.inf
                        
    """
    selection_type = {
        'SelectKBest' : SelectKBest(),#univariate selection
        'RFE' : RFE(estimator = SVR(kernel="linear")),#recursive feature extraction
        'SelectFromModel' : SelectFromModel(estimator = SVR(kernel="linear")),#feature importance
        'Correlation' : Corr_selection(),
        'PCA' : PCA()
    }
    select = selection_type[how]
    select.set_params(**params)
    select.fit(X,y)
    if how == 'PCA':
        features = [f'pca_{i}' for i in range(params['n_components'])]
    else:
        features = select.get_feature_names_out(input_features=X.columns)
    X = select.transform(X)
    return pd.DataFrame(X, columns=features)
