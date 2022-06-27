import pickle
import numpy as np

def load_data():
    file = open('training.pickle','rb')
    train_x, train_y = pickle.load(file)
    full_data = []
    for i in range(len(train_x)):
        full_data.append((train_x[i], train_y[i]))
    return np.array(full_data)
#print(train_x)
#print(train_y)