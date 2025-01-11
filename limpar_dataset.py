def correcao_nulos(dataset):
    #Completando os dois valores nulos de Embarked, dos Passageiros 62 e 830, com valores pesquisados da internet
    dataset.loc[dataset.Embarked.isna(),'Embarked'] = 'S'
    
    #Completando os valores nulos de Age, com a mediana do agrupamento de Pclass e Sex
    dataset['Age'] = dataset.Age.fillna(dataset.groupby(['Pclass','Sex'])['Age'].transform('median'))

    return dataset

def correcao_valores(dataset):
    
    #Trocar os valores de Survived (1 e 0) para True e False
    dataset['Survived'] = dataset.apply(lambda x: 'True' if x['Survived'] == 1 else 'False', axis=1)
    
    #Completar o valor (S,C,Q) de onde embarcou com os valores Southampton,Cherbourg,Queenstown
    dataset['Embarked'] = dataset.apply(lambda x: 'Southampton' if x['Embarked'] == 'S' \
                                                    else('Cherbourg' if x['Embarked'] == 'C' else 'Queenstown'),
                                                    axis= 1)
    
    #Trocar os valores de Pclass (1, 2 e 3) para 1st class, 2nd class e 3rd class
    dataset['Pclass'] = dataset.apply(lambda x: '1st class' if x['Pclass'] == 1 \
                                        else('2nd class' if x['Pclass'] == 2 else '3rd class'),
                                        axis=1)    
    
    #Criar uma nova coluna, AgeGroup, para colocar uma classificação de idade
    dataset['AgeGroup'] = dataset.apply(lambda x: 'Infant' if 0 < x['Age'] < 13        \
                                        else('Teenager' if 13 <= x['Age'] < 18         \
                                        else('Young adult' if 18 <= x['Age'] < 30      \
                                        else('Adult' if 30 <= x['Age'] < 60 else 'Elderly'))),
                                        axis=1)

    return dataset                                         