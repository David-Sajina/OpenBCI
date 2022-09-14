import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization
import os
import random
import time
from sklearn.model_selection import train_test_split


ACTIONS = ["krug", "kvadrat", "trokut"]

def create_data(starting_dir="alldata"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    # creating X, y
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "krug":
                combined_data.append([data, 0])

            elif action == "kvadrat":
                combined_data.append([data, 1])

            elif action == "trokut":
                combined_data.append([data, 2])


    print("length:", len(combined_data))
    return combined_data


print("creating training data")
traindata = create_data(starting_dir="alldata")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)



print(np.array(train_X).shape)
print(np.array(train_y).shape)
train_X = np.array(train_X)

train_y = np.array(train_y)
param = 90
train_X = np.array([train_X[n:n+param] for n in range(0, len(train_X), param)])
y = [train_y[i] for i in range(0, len(train_y), param)]
train_y = np.array(y)


indices = np.random.permutation(len(train_X))
train_X = train_X[indices]
train_y = train_y[indices]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, train_size=0.70)
print("shape:", X_train.shape)
print("shape:", y_train.shape)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0007,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)

model = Sequential()
model.add(Conv2D(16, (5, 5), padding="same", activation='relu', input_shape=train_X.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))

model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))


model.add(Conv2D(128, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(3, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

epochs = 10
mean = 0
batch_size = 32
model.summary()
for epoch in range(epochs):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.2)
    print((epoch+1)*10, "%")
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    mean += score[1]
    MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-{epochs}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
   # model.save(MODEL_NAME)

print(round(mean/10*100, 2),"%")
