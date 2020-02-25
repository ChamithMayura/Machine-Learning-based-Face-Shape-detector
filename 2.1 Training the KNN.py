import numpy as np

#import and load external data files
data= np.load('data.npy')
target= np.load('target.npy')


print(data.shape)
print(target.shape)


from sklearn.neighbors import KNeighborsClassifier

algorithm=KNeighborsClassifier()
algorithm.fit(data,target)

import joblib#save the trained algorithm with data

joblib.dump(algorithm,'KNN_model.sav')#save model in different file to maximuze the efficieny of the program.
