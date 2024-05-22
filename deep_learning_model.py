import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Read dataset from CSV file
df = pd.read_csv('dataset.csv')

# Separate features and labels
X = df['sentence'].tolist()
y = df['pos_tag'].tolist()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize input sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_sequence_length = max([len(x) for x in X])
X = pad_sequences(X, maxlen=max_sequence_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
embedding_dim = 50
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing set
y_pred = model.predict_classes(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
