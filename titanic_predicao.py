import pandas as pd
from limpar_dataset import correcao_nulos,correcao_valores
from sklearn.model_selection import train_test_split
from ajuste_hiper_modelos import fit_and_score,gaussian_cv,logistic_regression_cv,kneighbors_cv,random_forest_cv,linear_svc_cv
from plots import survived_embarked,survived_sex,survived_age_group,survived_social_class,plot_AUC,plot_rotulos,\
    plot_classif_report,plot_accuracy
from sklearn.metrics import classification_report

def main():
    #Reformulção do dataset
    '''dataset_limpo = limpar_dados()
    dataset_limpo.to_csv('Dataset/Titanic-Reformulado-Dataset.csv',index=False)'''

    #Mensurações,Treino,Teste e Avaliação dos modelos
    '''df = pd.read_csv('Dataset/Titanic-Reformulado-Dataset.csv')
    
    plots_mensuracoes(df)'''

    '''df_teste = pd.get_dummies(df[['Survived','Pclass','Sex','Embarked','AgeGroup']])

    X = df_teste.drop(['Survived','Sex_male'],axis=1)
    y = df['Survived']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)'''

    ##Avaliar modelos sem ajuste de hiperparâmetro
    '''print(fit_and_score(X_train,X_test,y_train,y_test))'''
    
    '''avaliar_modelos(X_train,X_test,y_train,y_test)'''

#Plots para mensurações
def plots_mensuracoes(df):
    survived_embarked(df)
    survived_sex(df)
    survived_age_group(df)
    survived_social_class(df)


#Avaliar os modelos com ajuste de hiperparâmetro
def avaliar_modelos(X_train,X_test,y_train,y_test):   
    models = {'GNB': gaussian_cv(X_train,y_train),
              'LR': logistic_regression_cv(X_train,y_train),
              'KN': kneighbors_cv(X_train,y_train),
              'RF': random_forest_cv(X_train,y_train),
              'LSVC': linear_svc_cv(X_train,y_train)}
    
    ##Plots do AUC
    '''for  in models.items():
        plot_AUC(nome_modelo,modelo,X_test,y_test)'''

    ##Plots de falso positivo/negativo
    '''for nome_modelo,modelo in models.items():
        plot_rotulos(nome_modelo,modelo,X_test,y_test)'''

    ##Plots de classification report
    '''metrics = ['precision','recall','f1-score','support']
    metrics_false = {name_model: [] for name_model in models.keys()}
    metrics_true = {name_model: [] for name_model in models.keys()}
    accuracy = {name_model: [] for name_model in models.keys()}
    
    for nome_mod,modelo in models.items():
        y_preds = modelo.predict(X_test)
        cr = classification_report(y_test,y_preds,output_dict=True)

        for metric_cr,valor_metric in cr.items():
            if metric_cr == 'False':
                for metric in metrics:
                    metrics_false[nome_mod].append(valor_metric.get(metric))
            
            elif metric_cr == 'True':
                for metric in metrics:
                    metrics_true[nome_mod].append(valor_metric.get(metric))

            elif metric_cr == 'accuracy':
                accuracy[nome_mod].append(valor_metric)

    
    df_false = pd.DataFrame(metrics_false,index=metrics).drop(['support']).T
    df_true = pd.DataFrame(metrics_true,index=metrics).drop(['support']).T
    df_accuracy = pd.DataFrame(accuracy,index=['accuracy'])

    plot_classif_report([df_true,df_false],metrics[:-1])
    plot_accuracy(df_accuracy,models.keys())'''

#Limpar e reformular o Dataset    
def limpar_dados():
    dataset_corregido = correcao_nulos(pd.read_csv('Dataset/Titanic-Dataset.csv'))
    dataset_corregido = correcao_valores(dataset_corregido)

    return dataset_corregido