import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import RocCurveDisplay,confusion_matrix

#Mensurações:
##Plot de sobreviventes pelo local de embarcação
def survived_embarked(df):
    pd.crosstab(df['Survived'],df['Embarked']).plot(kind='bar',figsize=[10,6])
    plt.title('Sobreviventes pela embarcagem')
    plt.xlabel('False = Não sobreviveu, True = sobreviveu')
    plt.ylabel('Amostragem')
    plt.legend(['Cherbourg','Queenstown','Southampton'])
    plt.savefig(fname='assets/mensuracoes/embarcacao.png',format='png')
    plt.show()

##Plot de sobreviventes pelo sexo
def survived_sex(df):
    pd.crosstab(df['Survived'],df['Sex']).plot(kind='bar',figsize=[10,6])
    plt.title('Sobriventes pelo sexo')
    plt.xlabel('False = Não sobreviveu, True = sobreviveu')
    plt.ylabel('Amostragem')
    plt.legend(['Mulher','Homem'])
    plt.savefig(fname='assets/mensuracoes/sexo.png',format='png')
    plt.show()

##Plot de sobreviventes pela faixa etária
def survived_age_group(df):
    pd.crosstab(df['AgeGroup'],df['Survived']).plot(kind='barh',figsize=[10,6])
    plt.title('Sobreviventes pela faixa etária')
    plt.xlabel('Amostragem')
    plt.ylabel('Faixa Etária')
    plt.legend(['Não sobreviveu','Sobreviveu'])
    plt.savefig(fname='assets/mensuracoes/faixa_etaria.png',format='png')
    plt.show()

##Plot de sobreviventes pela classe 
def survived_social_class(df):
    pd.crosstab(df['Pclass'],df['Survived']).plot(kind='barh',figsize=[10,6])
    plt.title('Sobreviventes pela classe')
    plt.xlabel('Amostragem')
    plt.ylabel('Classe ')
    plt.legend(['Não sobreviveu','Sobreviveu'])
    plt.savefig(fname='assets/mensuracoes/classe.png',format='png')
    plt.show()

#Plot de avaliação do AUC/Área Sob a Curva, probalidade do modelo de classificar corretamente um exemplo aleatório
def plot_AUC(nome_modelo,modelo,X_test,y_test):
    RocCurveDisplay.from_estimator(modelo,X_test,y_test)
    plt.savefig(fname=f'assets/avaliacoes_modelos/{nome_modelo}_AUC.png', format='png')
    plt.show()

#Plot de avaliação de falsos positivos e falsos negativos    
def plot_rotulos(nome_modelo,modelo,X_test,y_test):
    y_preds = modelo.predict(X_test)
    
    sns.set_theme(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                     annot=True,
                     cbar=False)
    
    plt.xlabel('Rótulos verdadeiros')
    plt.ylabel('Rótulos falso')
    plt.savefig(fname=f'assets/avaliacoes_modelos/{nome_modelo}_confusion_matrix.png', format='png')
    plt.show()

#Plot de avaliação do classification report    
def plot_classif_report(DataFrames,metrics):
    classes = ['True','False']
    for df,cls in zip(DataFrames,classes):
        fig,axes = plt.subplots(1,3, figsize=(15,6))
        for ax,metric in zip(axes,metrics):
            sns.barplot(data=df, x=df.index, y=metric, ax=ax)
            ax.set_title(f'{metric.capitalize()} por Modelo')
            ax.set_ylim(0,1.1)
            ax.set_xlabel('')
        
        fig.suptitle(cls)
        #plt.savefig(fname=f'assets/avaliacoes_modelos/{cls}_classification_report.png', format='png')
        plt.show()   

#Plot de accuracy dos modelos
def plot_accuracy(df,nome_modelos):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=df, ax=ax)
    ax.set_title('Accuracy por Modelo')
    ax.set_ylim(0,1.1)
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')

    plt.savefig(fname=f'assets/avaliacoes_modelos/accuracy.png', format='png')
    plt.show()