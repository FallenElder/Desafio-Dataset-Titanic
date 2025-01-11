from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

#Teste com os modelos, sem ajuste dos hiperparâmetros, com retorno do score de cada modelo
def fit_and_score(X_train,X_test,y_train,y_test):
    models = {'GNB': GaussianNB(),
              'Logistic Regression': LogisticRegression(),
              'KN': KNeighborsClassifier(),
              'RF': RandomForestClassifier(),
              'LSVC': svm.LinearSVC()}
    
    model_score = {}

    for name,model in models.items():
        model.fit(X_train,y_train)

        model_score[name] = model.score(X_test,y_test)
    
    return model_score


#Ajuste para GaussianNB
def gaussian_cv(X_train,y_train):
    gaussian = {'var_smoothing':np.logspace(-4,4,50)}

    rscv_gaussian = RandomizedSearchCV(GaussianNB(),
                                    param_distributions=gaussian,
                                    cv=5,
                                    n_iter=20)
    
    rscv_gaussian.fit(X_train,y_train)

    return rscv_gaussian


#Ajuste para LogisticRegression
def logistic_regression_cv(X_train,y_train):
    logreg_cv = {'C':np.logspace(1,20,50),
                   'solver':['liblinear'],
                   'penalty':['l2','l1']}
    
    rscv_log_reg = RandomizedSearchCV(LogisticRegression(),
                                    param_distributions=logreg_cv,
                                    cv=5,
                                    n_iter=20)
    
    rscv_log_reg.fit(X_train,y_train)

    return rscv_log_reg


#Ajuste para KNeighbors
def kneighbors_cv(X_train,y_train):
    knc = {'n_neighbors':range(1,21)}

    rscv_knc = RandomizedSearchCV(KNeighborsClassifier(),
                                    param_distributions=knc ,
                                    cv=5,
                                    n_iter=20)
    
    rscv_knc.fit(X_train,y_train)

    return rscv_knc


#Ajuste para RandomForest
def random_forest_cv(X_train,y_train):
    ranfor_cv = {'n_estimators': np.arange(100,1000,100),
                   'criterion': ['log_loss'],
                   'min_samples_split':np.arange(10,20,2),
                   'max_features':['log2'],
                   'min_samples_leaf': np.arange(1,5,1)}
    
    
    rscv_ran_for = RandomizedSearchCV(RandomForestClassifier(),
                                    param_distributions=ranfor_cv ,
                                    cv=5,
                                    n_iter=30)
    
    rscv_ran_for.fit(X_train,y_train)

    return rscv_ran_for


#Ajuste para LinearSVC
def linear_svc_cv(X_train,y_train):
    linear_svc_cv = {'C':np.logspace(1,20,50),
                      'penalty':['l2','l1'],
                      'dual':['auto',False],
                      'max_iter':np.arange(1000,1500,100)}
    
    rscv_linear_svc = RandomizedSearchCV(svm.LinearSVC(),
                                    param_distributions=linear_svc_cv,
                                    cv=5,
                                    n_iter=20)
    
    rscv_linear_svc.fit(X_train,y_train)

    return rscv_linear_svc