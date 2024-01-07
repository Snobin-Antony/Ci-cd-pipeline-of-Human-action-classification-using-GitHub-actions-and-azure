import unittest
import math
import argparse
from train import loadData, splitData, buildModel, assessModel
from sklearn.linear_model import LogisticRegression

import subprocess
# Install fsspec
subprocess.run(["pip", "install", "fsspec"])
# Now you can use fsspec or any other libraries that depend on it
import fsspec
# argparser = argparse.ArgumentParser()
# argparser.add_argument('--test', required=True, help='test data file to load')
# args = argparser.parse_args()

class TestTrain(unittest.TestCase):

    def test_loadData(self):
        #We know it should have 150 rows, so let's check that
        #We also know that the X and Y should be the same length
        X, Y = loadData("azureml://subscriptions/bf0717bf-dfd1-4019-a2b6-aa46e3899a4d/resourcegroups/assignment-snobin/workspaces/assignmentsnobin/datastores/workspaceblobstore/paths/UI/2024-01-06_061109_UTC/train.csv")
        self.assertGreaterEqual(len(X), 2056)
        self.assertEqual(len(Y), len(X))
        #We also know X should have two columns, so lets check that
        #   for the first entry
        self.assertEqual(len(X[0]), 561)
        return X, Y

    def setUp(self):
        self.testX, self.testY = self.test_loadData()   
        # self.testX = [[5.1,3.5], [4.9,3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6], [5.4, 3.9]]
        # self.testY = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-virginica']

    def test_splitData(self):
        #Test that we can split the data into train and test sets
        #We know that the train and test sets should add up to the same
        # length as the original data (i.e. we havn't lost anything)
        for i in range(1, 8):
            percentage_value  = i/10
            print("Testing with percentage", percentage_value)
            X_train, X_test, Y_train, Y_test = splitData(self.testX, self.testY, percentage_value)
            self.assertEqual(len(X_train) + len(X_test), len(self.testX))
            self.assertEqual(len(Y_train) + len(Y_test), len(self.testY))
            self.assertEqual(len(X_test), math.ceil(len(self.testX) * percentage_value))

    def test_buildModel(self):
        #Test the model builder returns a model of the correct type
        model = buildModel(self.testX, self.testY)
        self.assertIsInstance(model, LogisticRegression)

    def test_assessModel(self):
        #Test the accuracy function returns a value >=0 and <=1
        #Were giving the same test and train data which we shouldn't
        # but this is a test of its function not it's performance.
        model = buildModel(self.testX, self.testY)
        acc = assessModel(model, self.testX, self.testY)
        #TODO: finish these
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1) 
        print("Accuracy", acc)   
        # self.<FIXME>(acc, 0)
        # self.<FIXME>(acc, 1)

if __name__ == '__main__':
    unittest.main()