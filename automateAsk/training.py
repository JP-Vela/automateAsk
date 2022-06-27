import preprocessing as train_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_x, train_y = train_data.create_training_data()

num_samples, input_size = train_x.shape
num_epochs = 1000

print("\n")

model = Sequential([
    Dense(input_size, activation='relu'),
    Dense(16, activation='relu'),
    #Dropout(0.1, input_shape=(input_size,)),
    Dense(16, activation='relu'),
    Dense(len(train_data.classes), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fitness = model.fit(train_x,train_y,epochs=num_epochs,validation_data=(train_x,train_y),verbose=1)
model.save('models/autoAskV2')