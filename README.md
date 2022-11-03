# sentiment-analysis
Sentiment Analysis on US Airline Reviews

## ✨ Get Started
Instalar as dependências necessárias para iniciar

- **tensorflow**: Uma plataforma completa de código aberto para machine learning
- **keras**: API baseado no tensorflow que encapsula modelos de machine learning
- **pandas**: Ferramenta de código de aberto para análise e manipulação de dados
- **matplotlib**: Pacote para visualização de dados estático, dinâmico e/ou 3D

```pip
pip install tensorflow keras pandas matplotlib
```
Importar as bibliotecas devidademente

> OBS: Do keras, é necessário:
> - O interpretador NLP
> - O sequenciador
> - O modelo da rede neural
> - E as camadas da rede.

```py
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
```

## 🗃️ Framework OSEMN

OSEMN é um método de análise e processamento de dados bastante usado dentro do contexto de DataOps baseado na seguinte estrutura:
- `O` - Obtain: *Obter*
- `S` - Scrub: *Limpar*
- `E` - Explorer: *Explorar*
- `M` - Model: *Modelar*
- `N` - iNterpret: *Interpretar*

## 📚 Obter

Lê o dataset dentro do drive

```py
df = pd.read_csv("path/to/file/airline.csv")
```

## 🗑️ Limpar

Filtrar apenas as informaçõe de `texto` e `sentimento` do dataset

```py
tweet_df = df[['text','airline_sentiment']]
```

Remover mensagens com o sentimento `neutro` para classificação binária

```py
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
```

## 🔎 Explorar

Fatorizar os valores de sentimento

```py
sentiment_label = tweet_df.airline_sentiment.factorize()
```

Fatorizar os valores de texto

```py
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
```

## ⌛️ Modelar

Criar a rede neural do tipo `sequencial` (possui apenas um tensor de entrada e um tensor de sáida)

```py
model = Sequential()
```

Adicionar uma camada de `embedding`

```py
embedding_vector_length = 32
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
```

Adicionar uma camada de `dropout` para vetores unidimensionais

```py
model.add(SpatialDropout1D(0.25))
```

Adicionar uma camada de `Long Short-Term Memory` ou `LSTM` que escolhe como as camadas de dados serão processadas baseadas em como o `tensorflow` executa em memória

```py
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
```

Adicionar uma segunda camada de `dropout`, porém para valores puros

```py
model.add(Dropout(0.2))
```

Adicionar uma camada de `densidade`, que vai dar prioridade a ligações da rede que possuem mais força

```py
model.add(Dense(1, activation='sigmoid'))
```

Compilar a rede neural com todas as camadas montadas

```py
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
```

Iniciar o processo de treinamento da rede

```py
history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=5, batch_size=32)
```

## 📊 Interpretar

Plotar um gráfico contendo a curva de crescimento da precisão do modelo

```py
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy_Plot.jpg")
```
![image](https://user-images.githubusercontent.com/30305307/199713624-014568d5-0fbf-432b-8573-17fa8ec1adf7.png)

Plotar um gráfico contendo a curva de queda do índice de perda do modelo

```py
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss_Plot.jpg")
```
![image](https://user-images.githubusercontent.com/30305307/199713550-c1a67d4a-6ef9-4ccf-8d63-f5a979e24210.png)

Criar função para descrever o sentimento baseado o modelo treinado

```py
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    sentiment_response = sentiment_label[1][prediction]
    if sentiment_response == 'positive':
      print("😊")
    else:
      print("😡")
```

## 🌟 Veja a mágica acontecer

```py
predict_sentiment("I enjoyed my journey on this flight.")
```
> 😊


```py
predict_sentiment("This is the worst flight experience of my life!")
```
> 😡
