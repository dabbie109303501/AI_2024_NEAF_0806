# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io
# from os import walk
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix

# from tensorflow import keras
# from keras import layers

# #自行建立神經網路
# #準確度到 1
# #交叉驗證
# #建議用cnn

# #Load data
# train_x = []
# train_x_std = []
# train_y = []
# folder_name = ['Yes', 'No']

# #mkdir
# import os
# if not os.path.exists("./Data/Fig"):os.mkdir("./Data/Fig")

# i=0
# for folder in folder_name:
#     path = './Data/'+ str(folder) +'/'
#     for root, dirs, files in walk(path):
#         for f in files:
#             filename = path + f
#             # print(filename)
            
#             acc = scipy.io.loadmat(filename)
#             acc = acc['tsDS'][:,1].tolist()[0:7500]
#             train_x.append(acc)
#             train_x_std.append(np.std(acc))

#             if folder == 'Yes':    
#                 train_y.append(1)
#                 title = 'Original Signal With Chatter #'
#                 saved_file_name = './Data/Fig/Yes_'
            
#             if folder == 'No':
#                 train_y.append(0)
#                 title = 'Original Signal Without Chatter #'
#                 saved_file_name = './Data/Fig/No_'
#             '''
#             # plt.clf()
#             plt.figure(figsize=(7,4))
#             plt.plot(acc, 'b-', lw=1)
#             plt.title(title + str(i+1))
#             plt.xlabel('Samples')
#             plt.ylabel('Acceleration')
#             plt.savefig(saved_file_name + str(i+1) + '.png')                
#             # plt.show()
#             i = i + 1
#             '''

# train_x = np.array(train_x_std)
# train_y = np.array(train_y)

# scaler = MinMaxScaler(feature_range=(0,1)) #縮放到0~1
# train_x = scaler.fit_transform(train_x.reshape(-1,1)) #變直排
# # print(train_x)

# loo = LeaveOneOut() #交叉驗證

# #

# model = MLPClassifier(max_iter=500, batch_size=1, solver='adam')

# #
# y_pred = cross_val_predict(model, train_x, train_y , cv=loo)
# #  



# y_true = train_y

# print('Prediction: \t', y_pred)
# print('Ground Truth: \t', y_true)

# cf_m = confusion_matrix(y_true, y_pred)
# print('Confusion Matrix: \n',cf_m)

# tn,fp,fn,tp = cf_m.ravel()
# accuracy = (tn+tp)/(tn+tp+fn+fp)
# print('Accuracy: ',accuracy)  
     
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras import layers

#自行建立神經網路
#準確度到 1
#交叉驗證
#建議用cnn

#Load data
train_x = []
train_x_std = []
train_y = []
folder_name = ['Yes', 'No']

#mkdir
import os
if not os.path.exists("./Data/Fig"):os.mkdir("./Data/Fig")

i=0
for folder in folder_name:
    path = './Data/'+ str(folder) +'/'
    for root, dirs, files in walk(path):
        for f in files:
            filename = path + f
            # print(filename)
            
            acc = scipy.io.loadmat(filename)
            acc = acc['tsDS'][:,1].tolist()[0:7500]
            train_x.append(acc)
            train_x_std.append(np.std(acc))

            if folder == 'Yes':    
                train_y.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = './Data/Fig/Yes_'
            
            if folder == 'No':
                train_y.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = './Data/Fig/No_'
            '''
            # plt.clf()
            plt.figure(figsize=(7,4))
            plt.plot(acc, 'b-', lw=1)
            plt.title(title + str(i+1))
            plt.xlabel('Samples')
            plt.ylabel('Acceleration')
            plt.savefig(saved_file_name + str(i+1) + '.png')                
            # plt.show()
            i = i + 1
            '''

train_x = np.array(train_x_std)
train_y = np.array(train_y)

scaler = MinMaxScaler(feature_range=(0,1)) #縮放到0~1
train_x = scaler.fit_transform(train_x.reshape(-1,1)) #變直排
# print(train_x)

loo = LeaveOneOut() #交叉驗證

#

# model = MLPClassifier(max_iter=500, batch_size=1, solver='adam')

def create_cnn_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

loo = LeaveOneOut()
y_pred = []

for train_index, test_index in loo.split(train_x):
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]

    model = create_cnn_model()
    model.fit(X_train, y_train, epochs=1000, verbose=0)  # Increase epochs for higher accuracy

    prediction = model.predict(X_test)
    y_pred.append(int(prediction > 0.5))



y_true = train_y

print('Prediction: \t', y_pred)
print('Ground Truth: \t', y_true)

cf_m = confusion_matrix(y_true, y_pred)
print('Confusion Matrix: \n',cf_m)

tn,fp,fn,tp = cf_m.ravel()
accuracy = (tn+tp)/(tn+tp+fn+fp)
print('Accuracy: ',accuracy)  
     
