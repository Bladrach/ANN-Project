import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from time import time
from keras.models import load_model
import matplotlib.ticker as ticker
from matplotlib import cm


class TimingCallback(Callback):
    def __init__(self):
        self.logs = []

    def on_epoch_begin(self, epoch, logs=[]):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs=[]):
        self.logs.append(time()-self.starttime)

cb = TimingCallback()

df = pd.read_csv('mushrooms.csv')
dataset = df.values

le = LabelEncoder()
for i in range(len(dataset[0, :])):
    dataset[:, i] = le.fit_transform(dataset[:, i])

#print(df.head(5))

x = dataset[:, 1:].astype(int)
y = dataset[:, 0].astype(int)

x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size = 0.2)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5)

model = Sequential([
    Dense(30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22),
    Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')])

max_epoch = 15
batchSize = 16
learning_rate = 1e-3
decay_rate = learning_rate / max_epoch
momentum = 0.9
period_number = 3
sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)
#adam = Adam(lr = learning_rate)

model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('trained_model{epoch:d}.h5', save_weights_only = False, period = period_number)

hist = model.fit(x_train, y_train, 
                batch_size = batchSize, epochs = max_epoch, 
                validation_data = (x_val, y_val),
                callbacks = [checkpoint, cb])

elapsed_time = sum(cb.logs)

#print(cb.logs)
print('Elapsed time for training: {:.3f} seconds'.format(float(elapsed_time)))

print("Model is saved")
#print("Test Score: {:.3f}".format(float(100 * model.evaluate(x_test, y_test)[1])))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'], 'r-')
plt.title('Model loss (Elapsed time: {:.3f} seconds)'.format(float(elapsed_time)))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.grid()
plt.savefig("Loss_History.png")
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'], 'r-')
plt.title('Model accuracy (Elapsed time: {:.3f} seconds)'.format(float(elapsed_time)))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.grid()
plt.savefig("Train_Accuracy_History.png")
plt.show()

accuracy = list()
epoch = list()

for i in range(1, 6):
    model = load_model('trained_model' + str(i * period_number) + '.h5')
    score = model.evaluate(x_test, y_test)[1]
    print('Epoch {} test accuracy: {:.5f}'.format(str(i * period_number), float(score)))
    epoch.append(i*period_number)
    accuracy.append(score)

plt.plot(epoch, accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test'], loc = 'upper left')
plt.grid()
plt.savefig("Test_Accuracy_History.png")
plt.show()

# Confusion matrix
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
conf_mat = confusion_matrix(y_test, y_pred)

LABELS = ["Edible","Poisonous"]
fig, ax = plt.subplots()
im = ax.imshow(conf_mat, interpolation = 'nearest', cmap = plt.cm.Blues)
ax.figure.colorbar(im, ax = ax)

ax.set(yticks = [-0.5, 1.5], 
       xticks = [0, 1], 
       yticklabels = LABELS, 
       xticklabels = LABELS)

# should change to 
ax.yaxis.set_major_locator(ticker.IndexLocator(base = 1, offset = 0.5))
plt.savefig("Confusion_Matrix.png")
plt.show()
print(conf_mat)
