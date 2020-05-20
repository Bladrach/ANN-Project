import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
#from keras.layers import Dropout


df = pd.read_csv('mushrooms.csv')
dataset = df.values

le = LabelEncoder()
for i in range(len(dataset[0, :])):
    dataset[:, i] = le.fit_transform(dataset[:, i])

print(dataset.shape)

x = dataset[:, 1 :]
y = dataset[:, 0]

x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size = 0.2)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5)

model = Sequential([    
    Dense(48, kernel_initializer = 'uniform', activation ='relu', input_dim = 22),
    #Dropout(0.5),        
    Dense(1, kernel_initializer = 'uniform', activation ='sigmoid')])


model.compile(optimizer = 'sgd',loss = 'binary_crossentropy', metrics = ['accuracy'])

#to check if there is an improvement or not, so training will stop if it is not improving
early_stopping_monitor = EarlyStopping(patience = 3)

hist = model.fit(x_train, y_train, 
                batch_size = 16, epochs = 100, 
                validation_data = (x_val, y_val), 
                callbacks = [early_stopping_monitor])

model.save_weights("model.h5")
print("Model is saved")
print("Test Score: {}".format(float(model.evaluate(x_test, y_test)[1])))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'], 'r-')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'], 'r-')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()
