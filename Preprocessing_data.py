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


# *********************************

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data



pipe.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.
