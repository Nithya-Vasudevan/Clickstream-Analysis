# 93.004 %  accuracy for validation set
import pandas as pd

train = pd.read_csv("training_data.csv")
test = pd.read_csv("test_data .csv")
train = train[train['grp'].isin(test['grp'])] #196651
# Missing values of 'Ad' is replaced with 0's. (If 'Ad' is blank: then no ads clicked)
train['ad'] = train['ad'].fillna('0')
test['ad'] = test['ad'].fillna('0')

# Extracting hour from Timestamp
train = train.replace({'timestamp': r':..:...000-04:00$'},{'timestamp': ''}, regex=True)
train = train.replace({'timestamp': r'2018-04-30T'},{'timestamp': ''}, regex=True)
test = test.replace({'timestamp': r':..:...000-04:00$'},{'timestamp': ''}, regex=True)
test = test.replace({'timestamp': r'2018-04-30T'},{'timestamp': ''}, regex=True)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

X_labels = train.iloc[:,[1,2,3,6,7]].values
Y_labels = train.iloc[:,4].reshape(train.shape[0],1)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_labels[:,0] = le.fit_transform(train.ad)
X_labels[:,1] = le.fit_transform(train.link)
X_labels[:,2] = le.fit_transform(train.timestamp)
X_labels[:,3] = le.fit_transform(train.grp)
X_labels[:,4] = le.fit_transform(train.funnel_level)

#train.funnel_level = le.fit_transform(train.funnel_level)

ohe= preprocessing.OneHotEncoder(categorical_features = [2,3,4])
X_labels = ohe.fit_transform(X_labels).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_labels, Y_labels,test_size = 0.30, random_state = 42)

X_train = minmax.fit_transform(X_train)
X_test = minmax.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from sklearn.utils import class_weight
import numpy as np
from keras.callbacks import ReduceLROnPlateau

#Initialising the ANN
Nmodel = Sequential()

#Adding Dense Layers
Nmodel.add(Dense(units = X_labels.shape[1], activation = 'relu', input_dim=X_labels.shape[1]))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 100, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 70, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 50, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 30, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 15, activation = 'relu'))
Nmodel.add(Dense(units = Y_labels.shape[1], init = 'uniform', activation = 'sigmoid'))
Nmodel.summary()
#Compile the model
Nmodel.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics = ['accuracy'])

#Fitting the NN model

y_ints = [y.argmax() for y in y_train]
cw = class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)
cbks = [keras.callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
history = Nmodel.fit(X_train, y_train, epochs= 10, batch_size=1000,
                     callbacks=cbks, class_weight=cw,
                     shuffle = True)
#, validation_data=[X_test,y_test])

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['X_train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


xtest = test.iloc[:,[1,2,3,4,5]].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
xtest[:,0] = le.fit_transform(test.ad)
xtest[:,1] = le.fit_transform(test.link)

xtest[:,2] = le.fit_transform(test.timestamp)
xtest[:,3] = le.fit_transform(test.grp)
xtest[:,4] = le.fit_transform(test.funnel_level)

ohe= preprocessing.OneHotEncoder(categorical_features = [2,3,4])
xtest = ohe.fit_transform(xtest).toarray()
xtest = minmax.fit_transform(xtest)
y_prediction = Nmodel.predict_classes(xtest)

y_prediction.sum()

test1 = test
test1.insert(loc = 4, column = 'checkout', value = y_prediction)

test1.to_csv("test_withChkout1", index = False)

#part 2
X_labels = train.iloc[:,[1,2,3,4,6,7]].values
Y_labels = train.iloc[:,5].reshape(train.shape[0],1)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_labels[:,0] = le.fit_transform(train.ad)
X_labels[:,1] = le.fit_transform(train.link)
X_labels[:,2] = le.fit_transform(train.timestamp)
X_labels[:,4] = le.fit_transform(train.grp)
X_labels[:,5] = le.fit_transform(train.funnel_level)


ohe= preprocessing.OneHotEncoder(categorical_features = [2,4,5])
X_labels = ohe.fit_transform(X_labels).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_labels, Y_labels,test_size = 0.2899, random_state = 42)

X_train = minmax.fit_transform(X_train)
X_test = minmax.fit_transform(X_test)


Nmodel = Sequential()

#Adding Dense Layers
Nmodel.add(Dense(units = X_labels.shape[1], activation = 'relu', input_dim=X_labels.shape[1]))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 100, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 50, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 50, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 30, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 15, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 10, activation = 'relu'))
Nmodel.add(Dropout(0.05))
Nmodel.add(BatchNormalization())
Nmodel.add(Dense(units = 5, activation = 'relu'))

Nmodel.add(Dense(units = Y_labels.shape[1], activation = 'sigmoid'))
Nmodel.summary()
#Compile the model
Nmodel.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics = ['accuracy'])

#Fitting the NN model

y_ints = [y.argmax() for y in y_train]
cw = class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)
cbks = [keras.callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
history = Nmodel.fit(X_train, y_train, epochs= 25, batch_size=1000, callbacks=cbks, class_weight=cw,
                     shuffle = True), validation_data=[X_test,y_test])


xtest = test1.iloc[:,[1,2,3,4,5,6]].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
xtest[:,0] = le.fit_transform(test.ad)
xtest[:,1] = le.fit_transform(test.link)
xtest[:,2] = le.fit_transform(test.timestamp)
xtest[:,4] = le.fit_transform(test.grp)
xtest[:,5] = le.fit_transform(test.funnel_level)

ohe= preprocessing.OneHotEncoder(categorical_features = [2,4,5])
xtest = ohe.fit_transform(xtest).toarray()
xtest = minmax.fit_transform(xtest)
y_prediction = Nmodel.predict_classes(X_test)

#Precision, recall, f1 score, cm
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prediction)


from sklearn.metrics import precision_score, recall_score
prec = precision_score(y_test, y_prediction)
recal = recall_score(y_test, y_prediction)


y_prediction.sum()
Solution_frame = pd.DataFrame(y_prediction, columns= ['order_placed'])
testdf = pd.read_csv("test_data .csv")
delete = testdf.id
Solution_frame.insert(loc=0, column='id', value=delete)
Solution_frame.to_csv("4229.csv", index = False)


validate = pd.read_csv("test_data .csv")
vt = pd.read_csv("3973 93.csv")

validate.to_csv("final.csv", index= False)
validate.insert(loc=6, column = 'order_placed', value=vt.order_placed)
validate.insert(loc=6, column = 'checkout', value=test1.checkout)
validate = validate.replace({'timestamp': r':..:...000-04:00$'},{'timestamp': ''}, regex=True)
validate = validate.replace({'timestamp': r'2018-04-30T'},{'timestamp': ''}, regex=True)





#Task 2
#Q1
train.query('order_placed ==1').funnel_level.value_counts()

#Q2
Q2 = train.query('checkout == 1 & order_placed == 0')
Q21 = train.query('checkout == 0').append(train.query('order_placed == 1 & order_placed == 1'))
Que2 = Q2[~Q2['link'].isin(Q21['link'])]

Q2 = train.query('checkout == 1 & order_placed == 0').groupby(['link','checkout','order_placed']).size().reset_index()
Q21 = train.query('checkout == 0').groupby(['link','checkout','order_placed']).size().reset_index().append(train.query('order_placed == 1 & order_placed == 1').groupby(['link','checkout','order_placed']).size().reset_index())
Que2 = Q2[~Q2['link'].isin(Q21['link'])]

Que2 = Que2.rename(columns={0:'Freq_of_links'})
Que2.to_csv("Que2.csv", index=False)
#Q3
Q3 = train.query('~(order_placed == 1 or checkout == 1)').groupby(['link','checkout','order_placed']).size().reset_index()
Q31 = train.query('(order_placed == 1 or checkout == 1)').groupby(['link','checkout','order_placed']).size().reset_index()
Que3 = Q3[~Q3['link'].isin(Q31['link'])]
# Top of Que3 is the solution to Question 2.

Que3.to_csv("Que3.csv", index=False)

Q4 = Q4.query('funnel_level == "lower"').groupby(['funnel_level','grp']).size().reset_index()
Q41=Q4.query('funnel_level != "lower"').groupby(['funnel_level','grp']).size().reset_index()
Que4 = Q4[~Q4['grp'].isin(Q41['grp'])]

Que4.to_csv("Que4.csv", index = False)

#Additional inference
op1 = train.query('order_placed == 1 & checkout == 1')

op1grp = op1.groupby(['grp'])['id'].count()
op1grp = op1grp.reset_index()
op1grp = op1grp.rename(columns={'id':'freq'})
op1grp.to_csv("Additional_inference.csv", index= False)



adi = train.groupby(['grp', 'order_placed'])['id'].count().reset_index()

adipiv = adi.pivot(index='grp', columns='order_placed', values='id')
adipiv = adipiv.reset_index()
adipiv.to_csv("Additional Inference.csv", index=False)

