import numpy as np
from sklearn import neighbors

locusMutations = []
results = []

#import the CSV file with the genetic data
dataArray = np.genfromtxt('MOCK_DATA.csv', delimiter=',')
dataArray = dataArray[1:]

#format the genetic data into samples and results
for patient in dataArray:
    temp = []
    for locus in patient:
        temp.append(locus)
    lastElement = temp.pop(-1)
    locusMutations.append(temp)
    results.append(lastElement)

#remove data from whole set which we will use to test
#the remaining data is what we will use to train
test_locusMutations = []
test_results = []

#shuffle the overall, so we take different test value each time
np.random.shuffle(locusMutations)
np.random.shuffle(results)

i = 0
while i < 5:
    test_locusMutations.append(locusMutations.pop(i))
    test_results.append(results.pop(i))
    i = i+1

#Create and train a KNeighboursClassifier using the training data
classifier = neighbors.KNeighborsClassifier()
classifier.fit(locusMutations, results)

#Print what the results should be, followed by what the classifer predicts from testing data
print("The actual results should have been: ")
print (test_results)
print("Predicted results were: ")
print (classifier.predict(test_locusMutations))
