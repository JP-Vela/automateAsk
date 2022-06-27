import json
import math
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/autoAskV1')

file_name = "training.json"
file_data = {}

classes = []

with open(file_name) as json_file:
    file_data = json.load(json_file)
 
for data in file_data['training']:
     classes.append(data['description'])

print("\n")
print(f"Classes: {classes}")

result = model.predict([[1,0,1,1]])[0]
max_index = np.argmax(result)
confidence = math.floor(result[max_index]*100)
print(f"Result: {classes[max_index]} {confidence}%")
