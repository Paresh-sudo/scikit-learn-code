import sklearn.datasets as ds
import sklearn.preprocessing as pp
import numpy as np

# import numpy
# import array

iris = ds.load_iris()
iris_normalized = pp.normalize(iris.data,norm='l2')
print(iris_normalized.mean(axis=0))

#task2
enc = pp.OneHotEncoder()
iris_target_onehot = enc.fit_transform(iris.target.reshape(-1, 1))
print(iris_target_onehot.toarray()[[0,50,100]])

#task3
# iris.iloc[:50,:] = np.nan
# iris_imputed = pp.Imputer(missing_values=np.nan, strategy='mean')
# iris_imputed=iris_imputed.fit_transform(iris)
# # print(iris_imputed)
#print([0.75140029, 0.40517418, 0.45478362, 0.14107142])
