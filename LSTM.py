#1. Dataset Loading and Preprocessing:
import requests
url = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'
text = requests.get(url).text
print(text[:500])

import string
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
#print(text[:500])
print(len(text))

chars = sorted(list(set(text)))
char_to_int = {char: idx for idx, char in enumerate(chars)}
int_to_char = {idx: char for idx, char in enumerate(chars)}

tokenized = [char_to_int[c] for c in text]
print(tokenized[:50])

seq_length = 100
X = []
y = []

for i in range(len(text) - seq_length):
    input_seq = text[i:i + seq_length]
    output_char = text[i + seq_length]
    X.append([char_to_int[c] for c in input_seq])
    y.append(char_to_int[output_char])

import numpy as np 
from keras.utils import to_categorical
X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

#2. Build the LSTM model

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=50, input_length=seq_length))
model.add(LSTM(128))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#3. Train the model
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=2)
model.fit(X, y, batch_size=128, epochs=20, callbacks=[early_stop])

#4. Text Generation Function
def generate_text(seed_text, gen_length=500):
    generated = seed_text.lower()
    for _ in range(gen_length):
        # Convert seed to integers
        input_seq = [char_to_int.get(char, 0) for char in generated[-seq_length:]]
        input_seq = np.array(input_seq).reshape(1, -1)

        # Predict next character
        pred = model.predict(input_seq, verbose=0)
        next_index = np.argmax(pred)
        next_char = int_to_char[next_index]

        generated += next_char

    return generated

seed1 = "shall i compare thee to a summer's day"
seed2 = "to be or not to be that is the question"

print("Generated text from seed 1:")
print(generate_text(seed1))

print("\nGenerated text from seed 2:")
print(generate_text(seed2))