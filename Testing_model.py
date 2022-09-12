import numpy as np
import tensorflow as tf
import os


MODEL_NAME = 'new_models\\57.59-acc-10epoch-1662933517-loss-1.04.model'  # your model path here.

model = tf.keras.models.load_model(MODEL_NAME)

ACTIONS = ["kvadrat", "trokut"]


def create_data(starting_dir="test_dataset"):
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



traindata = create_data(starting_dir="test_dataset")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)

train_X = np.array(train_X)
train_y = np.array(train_y)
param = 90
train_X = np.array([train_X[n:n+param] for n in range(0, len(train_X), param)])
y = [train_y[i] for i in range(0, len(train_y), param)]
train_y = np.array(y)

#unison_shuffled_copies(train_X, train_y)
indices = np.random.permutation(len(train_X))
train_X = train_X[indices]
train_y = train_y[indices]

print("shape:", train_X.shape)
print("shape:", train_y.shape)

model.evaluate(train_X, train_y)
