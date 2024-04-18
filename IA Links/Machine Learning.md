

Sumário:
- Tensor: 
## Introdução

Criando um modelo e seu treinamento supervisionado. 
Será utilizado a biblioteca Sklearn

feature para treinos
```
porco1 = [0, 1, 0]

porco2 = [0, 1, 1]

porco3 = [1, 1, 0]

  

cachorro1 = [0, 1, 1]

cachorro2 = [1, 0, 1]

cachorro3 = [1, 1, 1]
```

Criando meus dados para treino, dividindo em features e classificações
```
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1, 1, 1, 0, 0, 0]
```

importando classe LinearSVC responsável por criar o modelo a ser treinado através do módulo svm
```
from sklearn.svm import LinearSVC
```

instanciando modelo e setando os dados com suas classes associadas: 0 - cachorro, 1 - porco
```
model = LinearSVC()
model.fit(dados, classes)
```

Criando vários animais misteriosos para verificar a resposta do modelo e prevendo qual animal baseado na informação dada ele, no caso um animal misterioso
A função predict retorna um array da biblioteca numpy
```
misterio1 = [1,1,1]

misterio2 = [1,1,0]

misterio3 = [0,1,1]

animais_misteriosos = [misterio1, misterio2, misterio3]

previsoes_animais_misteriosos  = model.predict(animais_misteriosos)
```

Definindo tipos de animais antecipadamente para a comparação com a previsão
```
animais_misteriosos_corretos = [0, 1, 1]
quantidade_animais_corretos = (previsoes_animais_misteriosos == animais_misteriosos_corretos).sum()
```

Verificando a taxa de acerto entre o que é esperado e as previsões
```
total = len(animais_misteriosos)
taxa_de_acerto = quantidade_animais_corretos/total
print("Taxa de acerto ", taxa_de_acerto *100)
```

Sklern já oferece funções para verificar a acurácia como o accuracy_score
```
from sklearn.metrics import accuracy_score
taxa_de_acerto = accuracy_score(animais_misteriosos_corretos, previsoes_animais_misteriosos)
print("Taxa de acerto", taxa_de_acerto * 100)
```

## Treinamento a partir de uma base de dados existente

Biblioteca responsável por ler CSV
```
import pandas as pd
```

Lendo a fonte de dados
```
uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

pd.read_csv(uri)
```

Lendo os 5 primeiros registros através do método head
```
dados = pd.read_csv(uri)

dados.head()
```

Renomeando o nome das colunas com o método rename, passando um dicionário com chave e valor
```
mapa = {

    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"

}

dados.rename(columns = mapa)
```

Segregando as feature que são as 3 primeiras colunas do resultados, que seria a ultima coluna
```
x = dados[["home","how_it_works","contact"]]
y = dados["bought"]

x.head()
```

Verificando quantos elementos tem dentro da fonte de dados através da propriedade Shape
```
dados.shape
```

Separando os dados de treinamentos dos de teste
```
treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
```

Instanciando o modelo de setando os dados de treinamento no mesmo
```
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
```

Passando os dados de teste e retornando as previsões do modelo
```
previsoes = modelo.predict(teste_x)
print(previsoes)
```

Obtendo a taxa de acerto do modelo a partir do método accuracy_score
```
# Obtendo a taxa de acerto do modelo a partir do método accuracy_score

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)
```

Utilizar funções nativas sklearn para a separação dos dados de teste e de treinamento
```
# Utilizar funções nativas sklearn para a separação dos dados de teste e de treinamento

from sklearn.model_selection import train_test_split
```

Definindo uma semente fixa para o algoritmo de separação não separar aleatoriamente a fonte de dados de treino e de teste
```
SEED = 20
```

Treinando o modelo a partir da função train_test_split, passando as feature (x), resultados (y), a semente (random_state = SEED), a proporção de teste (test_size = 0.25)
proporção de classificação para garantir que tenha quase a mesma quantidade de 
```
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
```

treinando e obtendo as previsões
```
modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)
```

Verificando a precisão das previsões
```
# Verificando a precisão das previsões

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acurácia foi %.2f%%" % acuracia)
```

## Deep learning

```
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
```

==Explicação==

`import pandas as pd`: **Pandas** é uma biblioteca de software criada para a linguagem Python para manipulação e análise de dados. Em particular, oferece estruturas de dados e operações para manipular tabelas numéricas e séries temporais. `pd` é um alias comum usado para pandas.

`from sklearn.model_selection import train_test_split`: **train_test_split** é uma função da biblioteca **scikit-learn**. É usada para dividir conjuntos de dados em dois grupos: treinamento e teste. O grupo de treinamento contém uma saída conhecida e o modelo aprende com esses dados para ser generalizado para outros dados posteriormente. Usamos o conjunto de testes para testar a precisão dos nossos modelos de aprendizado de máquina.


`import numpy as np`: **NumPy** é uma biblioteca para a linguagem Python, adicionando suporte para grandes matrizes e arrays multidimensionais, juntamente com uma grande coleção de funções matemáticas de alto nível para operar nesses arrays. `np` é um alias comum usado para NumPy.

`import seaborn as sns`: **Seaborn** é uma biblioteca de visualização de dados em Python baseada em matplotlib. Ela fornece uma interface de alto nível para desenhar gráficos estatísticos atraentes e informativos. `sns` é um alias comum usado para seaborn.

`from sklearn.metrics import confusion_matrix, accuracy_score`: **confusion_matrix** e **accuracy_score** são funções de avaliação de modelo da biblioteca **scikit-learn**. A matriz de confusão é usada para descrever o desempenho de um modelo de classificação. O score de acurácia é a proporção de previsões corretas feitas pelo modelo em todas as previsões feitas.

```
np.random.seed(123)
torch.manual_seed(123)
```

==Explicação==

`np.random.seed(123)`: Esta linha define a semente para o gerador de números aleatórios do **NumPy**. A semente é um número (ou vetor) usado para inicializar um gerador de números pseudoaleatórios. Se você usar a mesma semente, obterá a mesma sequência de números aleatórios toda vez que gerar números aleatórios. Isso é útil para a reprodutibilidade dos experimentos. É como se você desse uma “dica” para o computador gerar uma sequência de números aleatórios. Se você der a mesma “dica” (neste caso, 123), o computador vai gerar a mesma sequência de números aleatórios toda vez.

`torch.manual_seed(123)`: Similarmente, esta linha define a semente para o gerador de números aleatórios do **PyTorch**. Isso garante que os resultados sejam os mesmos toda vez que o código é executado, o que é especialmente importante ao treinar modelos de aprendizado profundo, onde a inicialização aleatória dos pesos pode levar a resultados diferentes a cada execução. É a mesma coisa que a linha acima, mas para a biblioteca **PyTorch**, que é usada principalmente para aprendizado profundo.

Leitura de csv 
```
previsores = pd.read_csv('/content/entradas_breast.csv')
classe = pd.read_csv('/content/saidas_breast.csv')
```

Divisão dos dados para treinamento e para teste
`previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)`

==Explicação:==

`previsores_treinamento, previsores_teste`: Essas são as variáveis que vão guardar os dados de entrada (ou previsores) que serão usados para treinar o modelo e para testá-lo, respectivamente.

`classe_treinamento, classe_teste`: Essas variáveis vão guardar os dados de saída (ou classes) correspondentes aos previsores de treinamento e teste.

`train_test_split(previsores, classe, test_size = 0.25)`: Aqui estamos chamando a função `train_test_split` e passando os dados de entrada (`previsores`), os dados de saída (`classe`) e o tamanho do conjunto de teste (`test_size = 0.25`). O `test_size = 0.25` significa que 25% dos dados serão usados para teste e os outros 75% para treinamento.

Uma parte para “ensinar” ao computador (chamada de dados de treinamento).
Outra parte para testar se o computador aprendeu direito (chamada de dados de teste).

`previsores_treinamento` e `classe_treinamento` são os dados para ensinar. `previsores_teste` e `classe_teste` são os dados para testar.

```
previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype=torch.float)

classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype = torch.float)
```

==Explicação==

`previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype=torch.float)`

`np.array(previsores_treinamento)`: Aqui, estamos convertendo a lista `previsores_treinamento` em uma matriz NumPy. Isso é útil porque o PyTorch trabalha bem com tensores, e os arrays NumPy podem ser facilmente convertidos em tensores.

`torch.tensor(...)`: Em seguida, estamos criando um tensor do PyTorch a partir da matriz NumPy criada anteriormente. Um tensor é uma estrutura de dados similar a uma matriz, mas com capacidade de ser utilizada em GPUs para cálculos paralelos, o que é útil para acelerar operações em aprendizado de máquina.

`dtype=torch.float`: Estamos especificando que os números no tensor devem ser do tipo ponto flutuante (float). Isso é importante porque muitos algoritmos de aprendizado de máquina funcionam melhor com números decimais em vez de inteiros.

`classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype=torch.float)`: Similar ao primeiro passo, aqui estamos convertendo a lista `classe_treinamento` em uma matriz NumPy e depois em um tensor do PyTorch, ambos com tipo de dados ponto flutuante.

Resumindo, essas duas linhas de código convertem os dados de treinamento em tensores do PyTorch, que são estruturas de dados eficientes para trabalhar com algoritmos de aprendizado de máquina, especialmente quando se trata de realizar cálculos em GPUs.

```
dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)
```

==Expliação==

Este trecho de código está criando um objeto `TensorDataset` a partir de dois tensores: `previsores_treinamento` e `classe_treinamento`. Esse trecho de código está basicamente preparando seus dados para o treinamento de um modelo de aprendizado de máquina.

**TensorDataset** é uma classe utilitária do PyTorch, especificamente do módulo `torch.utils.data`. Ela é usada para encapsular conjuntos de dados onde cada amostra é representada por um par de tensores correspondentes. Neste caso, `previsores_treinamento` e `classe_treinamento`. 

`previsores_treinamento`: Este tensor contém os dados de entrada (ou seja, os recursos) para o treinamento do modelo de aprendizado de máquina. Cada linha deste tensor representa uma amostra de treinamento e cada coluna representa um recurso diferente. são os dados de entrada, como informações sobre um conjunto de imagens.

`classe_treinamento`: Este tensor contém as etiquetas ou respostas correspondentes para cada amostra de treinamento em `previsores_treinamento`.  são as respostas corretas, como se a imagem é de um gato ou cachorro.

O objeto `TensorDataset` resultante, `dataset`, pode ser usado para criar um objeto `DataLoader`, que permite carregar os dados em lotes e realizar outras operações úteis, como embaralhamento e carregamento paralelo dos dados. Isso é especialmente útil ao treinar modelos de aprendizado de máquina em grandes conjuntos de dados.

```
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
```

==Explicação==

Este trecho de código está criando um objeto `DataLoader` a partir do `dataset` que você criou anteriormente.

**DataLoader** é uma classe do PyTorch, especificamente do módulo `torch.utils.data`. Ela é usada para carregar os dados em lotes durante o treinamento de um modelo de aprendizado de máquina. Aqui estão os detalhes dos parâmetros:

`dataset`: Este é o conjunto de dados que você deseja carregar. No seu caso, é o `TensorDataset` que você criou anteriormente.

`batch_size=10`: Este é o número de amostras por lote. O modelo será atualizado após cada lote de 10 amostras. significa que ele vai pegar 10 exemplos de cada vez para treinar o modelo.

`shuffle=True`: Isso significa que as amostras serão embaralhadas a cada época (uma passagem completa através de todo o conjunto de dados). Isso é útil para garantir que o modelo não aprenda a ordem dos dados e possa generalizar bem para dados não vistos. significa que ele vai misturar os exemplos para que o modelo não aprenda a ordem dos dados.

Então, `train_loader` será usado para iterar sobre o conjunto de dados em lotes durante o treinamento do seu modelo de aprendizado de máquina.

```
# 30 -> 16 -> 16 -> 1
# (entradas + saida) / 2 = (30 + 1) / 2 = 16

classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
```

==Explicação==

Este é um exemplo de uma rede neural simples implementada usando a biblioteca PyTorch. A rede é composta por três camadas totalmente conectadas (ou densas), cada uma seguida por uma função de ativação ReLU, exceto a última que é seguida por uma função de ativação Sigmoid.

`nn.Linear(in_features=30, out_features=16)`: Esta é a primeira camada da rede neural. É uma camada linear (também conhecida como totalmente conectada ou densa) que recebe 30 recursos de entrada e retorna 16 saídas. Cada neurônio nesta camada está conectado a todos os 30 neurônios da camada anterior.

`nn.ReLU()`: Esta é a função de ativação ReLU (Rectified Linear Unit). É usada para adicionar não-linearidade ao modelo. A função ReLU é comumente usada em redes neurais e retorna zero para qualquer entrada negativa, mas para qualquer entrada positiva x, retorna esse mesmo valor x.

`nn.Linear(16, 16)`: Esta é a segunda camada da rede neural. É semelhante à primeira camada, mas desta vez, recebe 16 recursos de entrada (que são as saídas da primeira camada) e também retorna 16 saídas.

`nn.ReLU()`: Outra função de ativação ReLU é aplicada.

`nn.Linear(16, 1)`: Esta é a terceira e última camada da rede neural. Recebe 16 recursos de entrada e retorna 1 saída.

`nn.Sigmoid()`: Finalmente, a função de ativação Sigmoid é aplicada. Esta função mapeia a saída final para um intervalo entre 0 e 1, o que pode ser útil para problemas de classificação binária.

Os comentários no código indicam a estrutura da rede neural: “30 -> 16 -> 16 -> 1” significa que a rede tem uma camada de entrada com 30 neurônios, duas camadas ocultas com 16 neurônios cada e uma camada de saída com 1 neurônio.

O comentário “(entradas + saída) / 2 = (30 + 1) / 2 = 16” é uma regra prática comum para escolher o número de neurônios nas camadas ocultas. No entanto, este é apenas um ponto de partida e o número ideal de neurônios pode depender de muitos fatores e geralmente requer experimentação e ajuste.

```
criterion = nn.BCELoss()
```

==Explicação==

`nn.BCELoss()` é a abreviação de Binary Cross Entropy Loss. É uma função de perda comumente usada em problemas de classificação binária, ou seja, quando a saída esperada é 0 ou 1.

Ela mede a diferença entre a saída prevista pela rede neural e a saída real (ou seja, o rótulo ou a verdade do chão). Durante o treinamento, o objetivo é minimizar essa perda, ajustando os pesos da rede neural. Isso ajuda a rede a aprender a fazer previsões mais precisas. É uma forma de medir o quão bem a rede neural está fazendo durante o treinamento. Se a rede errar, essa medida aumenta. Se acertar, ela diminui. O objetivo é fazer essa medida ficar o menor possível ajustando a rede. Isso ajuda a rede a aprender a fazer previsões melhores.

```
optimizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.0001)
```

==Explicação==

`torch.optim.Adam`: Isso se refere ao otimizador Adam, que é um método de otimização de gradiente descendente estocástico proposto por Kingma e Ba. É conhecido por ser eficaz na prática e requer pouca configuração de hiperparâmetros. É um tipo de otimizador que ajuda a rede neural a aprender mais rápido e de maneira mais eficiente.

`classificador.parameters()`: Aqui, `classificador` é provavelmente uma rede neural definida em PyTorch. O método `.parameters()` retorna um iterador dos parâmetros (pesos e bias) da rede que serão otimizados. São as coisas que o otimizador vai ajustar para melhorar a rede neural.

`lr=0.001`: `lr` é a taxa de aprendizado. É um hiperparâmetro que determina o tamanho do passo que o otimizador dará para alcançar o mínimo. Uma taxa de aprendizado muito alta pode fazer o modelo convergir muito rapidamente, mas também pode oscilar em torno do mínimo. Uma taxa de aprendizado muito baixa pode fazer o modelo convergir muito lentamente. É o quão rápido a rede neural vai aprender.
 
 `weight_decay=0.0001`: O decaimento de peso é uma técnica de regularização que impede que os pesos na rede neural fiquem muito grandes adicionando uma penalidade adicional (proporcional ao tamanho dos pesos) à função de perda. É uma técnica para evitar que a rede neural aprenda demais de uma só vez, o que pode causar problemas.