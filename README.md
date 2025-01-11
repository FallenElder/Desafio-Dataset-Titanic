# Desafio Dataset Titanic

## Objetivo

Este projeto foi feito com intuito de ser um teste e desafio, para desenvolver uma experiência inicial de DataScience e compreender de forma amador um projeto simples.

**Foco do projeto:**

- Prever a probabilidade de sobrevivência
- Analisar alguns fatores que influenciaram a sobrevivência
- Identificar os grupos de passageiros mais vulneráveis
- Comparar diferentes algoritmos de machine learning

## Tecnologias Usadas

### Back-End

- Python
- Pandas
- Scikit-Learn
- Numpy

### Plotagem

- Matplotlib
- Seaborn

### Modelos/Algoritmos

- Random Forest
- Logistic Regression
- Gaussian Naïve Bayes/GaussianNB
- KNeighbors
- Support Vector Machine/SVM

## Mensurações

### Sobrevivência pelos fatores [Classe, Embarcação, Faixa Etária e Sexo/Gênero]

![sobrevivente_por_classe](assets/mensuracoes/classe.png)
![sobrevivente_por_embarcacao](assets/mensuracoes/embarcacao.png)
![sobrevivente_por_faixa_etaria](assets/mensuracoes/faixa_etaria.png)
![sobrevivente_por_sexo](assets/mensuracoes/sexo.png)

## Avaliações dos Modelos

### Gráficos para cada Modelo

- AUC/Area Under the Curve
- Matriz de confusão

### Random Forest

![AUC_RF](assets/avaliacoes_modelos/Redimensionado/RF_AUC.png) ![CM_RF](assets/avaliacoes_modelos/Redimensionado/RF_confusion_matrix.png)

### Logistic Regression

![AUC_LR](assets/avaliacoes_modelos/Redimensionado/LR_AUC.png) ![CM_LR](assets/avaliacoes_modelos/Redimensionado/LR_confusion_matrix.png)

### GaussianNB

![AUC_GaussianNB](assets/avaliacoes_modelos/Redimensionado/GNB_AUC.png) ![CM_GaussianNB](assets/avaliacoes_modelos/Redimensionado/GNB_confusion_matrix.png)

### KNeighbors

![AUC_KN](assets/avaliacoes_modelos/Redimensionado/KN_AUC.png) ![CM_KN](assets/avaliacoes_modelos/Redimensionado/KN_confusion_matrix.png)

### SVM/LinearSVC

![AUC_LSVC](assets/avaliacoes_modelos/Redimensionado/LSVC_AUC.png) ![CM_LSVC](assets/avaliacoes_modelos/Redimensionado/LSVC_confusion_matrix.png)

### Gráficos Finais

- Classification Report entre os Modelos (Sobreviveu ou Não Sobreviveu - True ou False)
- Accuracy entre os Modelos

**Modelos:**

- GNB - GaussianNB
- LR - Logistic Regression
- KN - KNeighbors
- RF - Random Forest
- LSVC - LinearSVC/SVM

### Classification Report

![classification_report_true](assets/avaliacoes_modelos/True_classification_report.png) ![classification_report_false](assets/avaliacoes_modelos/False_classification_report.png)

### Accuracy

![accuracy_models](assets/avaliacoes_modelos/accuracy.png)

## Material Usado

Dataset Titanic - <https://www.kaggle.com/datasets/yasserh/titanic-dataset>
