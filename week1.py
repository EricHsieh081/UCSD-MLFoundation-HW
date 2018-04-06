import numpy as np
import math

#Part1 Nearest Neighboring Classification
#==================================================================
# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('column_3C.dat', converters={6: lambda s: labels.index(s)} )

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

# Modify this Cell
import math

def distance(v1, v2):
    return math.sqrt(np.sum((v1-v2)**2))

def classify(trainx, curx):
    minIndex = 0
    minDis = distance(trainx[0], curx)
    for i in range(len(trainx)):
        curDis = distance(trainx[i], curx)
        if (curDis < minDis):
            minDis = curDis
            minIndex = i
    return minIndex

def NN_L2(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy 
    ### BEGIN SOLUTION
    predict = np.zeros(len(testx))
    for i in range(len(testx)):
        index = classify(trainx, testx[i])
        predict[i] = trainy[index]

    return predict
    ### END SOLUTION

testy_L2 = NN_L2(trainx, trainy, testx)
assert( type( testy_L2).__name__ == 'ndarray' )
assert( len(testy_L2) == 62 ) 
assert( np.all( testy_L2[50:60] == [ 0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.] )  )
assert( np.all( testy_L2[0:10] == [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.] ) )
#==================================================================

# Below shows the L1 Form distance

def distanceL1(v1, v2):
    return (np.sum(abs(v1-v2)))

def classifyL1(trainx, curx):
    minIndex = 0
    minDis = distanceL1(trainx[0], curx)
    for i in range(len(trainx)):
        curDis = distanceL1(trainx[i], curx)
        if (curDis < minDis):
            minDis = curDis
            minIndex = i
    return minIndex

def NN_L1(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy 
    
    ### BEGIN SOLUTION
    predict = np.zeros(len(testx))
    for i in range(len(testx)):
        index = classifyL1(trainx, testx[i])
        predict[i] = trainy[index]
    print(predict)
    return predict   
    ### END SOLUTION