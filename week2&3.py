# Standard includes
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
# Useful module for dealing with the Gaussian density
from scipy.stats import norm, multivariate_normal 

# Load data set.
data = np.loadtxt('wine.data.txt', delimiter=',')
# Names of features
featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                'OD280/OD315 of diluted wines', 'Proline']
# Split 178 instances into training set (trainx, trainy) of size 130 and test set (testx, testy) of size 48
np.random.seed(0)
perm = np.random.permutation(178)
trainx = data[perm[0:130],1:14]
trainy = data[perm[0:130],0]
testx = data[perm[130:178], 1:14]
testy = data[perm[130:178],0]

def fit_generative_model(x,y):
    k = 3  # labels 1,2,...,k
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k+1,d))
    sigma = np.zeros((k+1,d,d))
    pi = np.zeros(k+1)
    for label in range(1,k+1):
        indices = (y == label)
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1)
        pi[label] = float(sum(indices))/float(len(y))
    return mu, sigma, pi

# Fit a Gaussian generative model to the training data
mu, sigma, pi = fit_generative_model(trainx,trainy)
print(sigma)

# Now test the performance of a predictor based on a subset of features
def test_model(mu, sigma, pi, features, tx, ty):
    ###
    k = 3
    size = len(ty)
    points = np.zeros((size, k+1))
    for test_i in range(0, size):
        for label in range(1, k+1):
            tarCov = np.zeros((len(features), len(features)))
            for i in range(0, len(features)):
                for j in range(0, len(features)):
                    tarCov[i, j] = sigma[label, features[i], features[j]]
            points[test_i][label] = np.log(pi[label]) + multivariate_normal.logpdf(tx[test_i, features], mean=mu[label, features], cov=tarCov)
    predictions = np.argmax(points[:,1:4], axis=1) +1 #add 1 is to match the label
    errors = np.sum(predictions != testy)
    print(str(errors)+"/"+str(size))

test_model(mu, sigma, pi, [2, 4, 6], testx, testy) #changable features 
###