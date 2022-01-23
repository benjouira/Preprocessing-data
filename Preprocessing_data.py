from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)

print(scaler.mean_)
     
print(scaler.scale_)

X_scaled = scaler.transform(X_train)
X_scaled


print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
