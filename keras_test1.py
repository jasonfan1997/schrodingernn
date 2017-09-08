# LSTM and CNN for sequence classification
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

def standardize_data(array):
    #takes in 2d arrays
    #relative scale
    a = array.copy()
    for i in range(a.shape[-1]):
        mean = np.mean(a[:, i])
        std = np.std(a[:, i])
        a[:,i] = (a[:,i] - mean)/std
    return a

# load the dataset
DIR = "../data/stock_train_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read 88 features, weight, label
all_set = pd.read_csv(DIR, skipinitialspace=True, skiprows=0, usecols=COLUMNS).as_matrix()
np.random.shuffle(all_set)

TESTDIR="../data/stock_test_data_20170901.csv"
pred_col=list(range(1,89))   #1-88,Features
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True, skiprows=0, usecols=pred_col).as_matrix()

#fea_col = training_set[...,:88]
#label_col = [...,-1]

cut = math.floor(all_set.shape[0]*0.7)
X_train=all_set[0:cut, :88]
y_train=all_set[0:cut, -1]
training_weight=all_set[0:cut,-2]
X_test = all_set[cut:, :88]
y_test = all_set[cut:, -1]

#standardize the data
X_train=standardize_data(X_train)
X_test=standardize_data(X_test)

# create the model
model = Sequential()
model.add()
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



model.fit(X_train, y_train, epochs=1, batch_size=1000, validation_data=(X_test,y_test), sample_weight=training_weight) #not sure whether validation_split uses weight
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#model.save('test_1.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model identical to the previous one
#model = keras.models.load_model('my_model.h5')