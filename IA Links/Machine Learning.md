
## Introdução

Criando um modelo e seu treinamento supervisionado. 
Será utilizado a biblioteca Sklearn


```
# feature para treino

porco1 = [0, 1, 0]

porco2 = [0, 1, 1]

porco3 = [1, 1, 0]

  

cachorro1 = [0, 1, 1]

cachorro2 = [1, 0, 1]

cachorro3 = [1, 1, 1]
```

```
# Criando meus dados para treino

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

  

# Classificando através de classes

classes = [1, 1, 1, 0, 0, 0]
```

```
# importando classe LinearSVC responsável por criar o modelo a ser treinado através do módulo svm

from sklearn.svm import LinearSVC
```

```
# instanciando modelo

model = LinearSVC()

  

# setando os dados com suas classes associadas: 0 - cachorro, 1 - porco

model.fit(dados, classes)
```

```
# Criando vários animais misteriosos para verificar a resposta do modelo

misterio1 = [1,1,1]

misterio2 = [1,1,0]

misterio3 = [0,1,1]

  

animais_misteriosos = [misterio1, misterio2, misterio3]

  

# Prevendo qual animal baseado na informação dada ele, no caso um animal misterioso

# A função predict retorna um array da biblioteca numpy

previsoes_animais_misteriosos  = model.predict(animais_misteriosos)
```

```
# Definindo tipos de animais antecipadamente para a comparação com a previsão

animais_misteriosos_corretos = [0, 1, 1]

  

quantidade_animais_corretos = (previsoes_animais_misteriosos == animais_misteriosos_corretos).sum()
```

```
# Verificando a taxa de acerto entre o que é esperado e as previsões

total = len(animais_misteriosos)

taxa_de_acerto = quantidade_animais_corretos/total

print("Taxa de acerto ", taxa_de_acerto *100)
```

```
# Sklern já oferece funções para verificar a acurácia como o accuracy_score

from sklearn.metrics import accuracy_score
```

```
taxa_de_acerto = accuracy_score(animais_misteriosos_corretos, previsoes_animais_misteriosos)

print("Taxa de acerto", taxa_de_acerto * 100)
```


## Treinamento a partir de uma base de dados existente


```
# Biblioteca responsável por ler CSV

import pandas as pd
```

```
# Lendo a fonte de dados

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

pd.read_csv(uri)
```

```
# Lendo os 5 primeiros através do método head

dados = pd.read_csv(uri)

dados.head()
```

```
# renomeando o nome das colunas com o método rename, passando um dicionario com chave e valor

mapa = {

    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"

}

dados.rename(columns = mapa)
```

```
# segregando as feature que são as 3 primeiras colunas dos resultados, que seri a ultima coluna

x = dados[["home","how_it_works","contact"]]
y = dados["bought"]

x.head()
```

```
# Verificando quantos elementos tem dentro da fonte de dados através da propriedade Shape

dados.shape
```

```
# Separando os dados de treinamento dos de teste

treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
```

```
# Instanciando o modelo e setando os dados de treinamento no mesmo

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
```

```
# Passando os dados de teste e retornando as previsões do modelo

previsoes = modelo.predict(teste_x)
print(previsoes)
```

```
# Obtendo a taxa de acerto do modelo a partir do método accuracy_score

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)
```

```
# Utilizar funções nativas sklearn para a separação dos dados de teste e de treinamento

from sklearn.model_selection import train_test_split
```

```
# Definindo uma semente fixa para o algitmo de separação não separar aleatoriamente a fonte de dados de treino e de teste

SEED = 20
```

```
# Treinando o modelo a partir da função train_test_split, passando as feature (x), resultados (y), a semente (random_state = SEED), a proporção de teste (test_size = 0.25)

# proporção de classificação para garantir que tenha quase a mesma quantidade de classe no treinamento quanto no teste (stratify = y)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
```

```
# treinando e obtendo as previsões

modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)
```

```
# Verificando a precisão das previsões

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acurácia foi %.2f%%" % acuracia)
```