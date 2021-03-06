import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization
import os
import random
import time


ACTIONS = ["krug", "kvadrat", "trokut"]
#reshape = (-1, 16, 60)

def create_data(starting_dir="data"):
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
        #np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
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
                #np.append(combined_data, np.array([data, [1, 0]]))
                combined_data.append([data, 1])

            elif action == "trokut":
                combined_data.append([data, 2])

   # np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data


print("creating training data")
traindata = create_data(starting_dir="data")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)

print("creating testing data")
testdata = create_data(starting_dir="validation_data")
test_X = []
test_y = []
for X, y in testdata:
    test_X.append(X)
    test_y.append(y)

print(len(train_X))
print(len(test_X))


print(np.array(train_X).shape)
print(np.array(train_y).shape)
train_X = np.array(train_X)#.reshape(reshape)
test_X = np.array(test_X)#.reshape(reshape)
train_y = np.array(train_y)
test_y = np.array(test_y)
param = 90
train_X = np.array([train_X[n:n+param] for n in range(0, len(train_X), param)])
y = [train_y[i] for i in range(0, len(train_y), param)]
train_y = np.array(y)
test_X = np.array([test_X[n:n+param] for n in range(0, len(test_X), param)])
y = [test_y[i] for i in range(0, len(test_y), param)]
test_y = np.array(y)

print(train_X.shape)
print(train_y.shape)
# samples, rows, cols, channels


model = Sequential()
model.add(Conv2D(32, (5, 5), padding="same", activation='relu', input_shape=train_X.shape[1:]))
model.add(MaxPooling2D((5, 5), padding="same"))
model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
model.add(MaxPooling2D((5, 5), padding="same"))
model.add(Conv2D(128, (5, 5), padding="same", activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(3, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
indices = np.random.permutation(len(train_X))
train_X = train_X[indices]
train_y = train_y[indices]
epochs = 10
batch_size = 32
model.summary()
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    print((epoch+1)*10, "%")
    #score = model.evaluate(test_X, test_y, batch_size=batch_size)
    #print(score)
    #MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
    #model.save(MODEL_NAME)
#print("saved:")
#print(MODEL_NAME)


