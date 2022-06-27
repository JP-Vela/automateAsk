import json
import numpy as np
import sklearn

file_name = "training.json"
file_data = {}

with open(file_name) as json_file:
    file_data = json.load(json_file)
 

#print(file_data["training"][0])

train_x = []
train_y = []
classes = []

def create_training_data():
    global train_x
    global train_y
    global classes

    #Set training samples as an array
    for data in file_data['training']:
        for sample in data['data']:
            train_x.append(sample)


    print(f"Num Samples:{len(train_x)}\n") #number of samples (should be 16 because 2x2x2x2=16)

    #Set training labels as an array
    for data in file_data['training']:
            for i in range(len(data['data'])):
                train_y.append(data['label'])

    print(f"Num Labels:{len(train_y)}\n") #number of labels (should be 4 because lamp on, lights off, and lights off blinds open)

    #Set classes as an array
    for data in file_data['training']:
        classes.append(data['label'])

    print(f"Num Classes:{len(classes)}\n")


    print("Samples \t Labels")

    #convert to numpy arrays
    classes = np.array(classes)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    for i in range(len(train_y)):
        print(f"{train_x[i]}\t{train_y[i]}")

    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
    return (train_x, train_y)