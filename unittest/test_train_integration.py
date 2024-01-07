import unittest
import tempfile
import os
import mlflow
import argparse
from train import trainModel
from sklearn.linear_model import LogisticRegression

# argparser = argparse.ArgumentParser()
# argparser.add_argument('--train', required=True, help='train data file to load')
# args = argparser.parse_args()

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.classes = ['LAYING', 'STANDING', 'WALKING', 'SITTING']
        self.numFeatures = 561

    def test_pipelineToModel(self):
        #We're going to start saving and loading stuff here and we don't
        # want to pollute the filesystem so we'll use a temporary directory
        # that will get cleaned up when we're done.
        # Python has a nice context manager for this in the tempfile module.
        with tempfile.TemporaryDirectory() as tmpdir:
            #Use the trainmodel function to create a model
            model, trainX, trainY = trainModel("train.csv", tmpdir + "/model")
            #Check the training data is as expected
            #Check X and Y are the same length
            self.assertEqual(len(trainX), len(trainY))
            #Check *all* the rows match what we expect
            for i in range(len(trainX)):
                #Check it has the right number of features
                self.assertEqual(len(trainX[i]), self.numFeatures)
                #Check it's one of the allowed values
                self.assertIn(trainY[i], self.classes)
                #Check if we predict it we get a allowed value from the model!
                self.assertIn(model.predict([trainX[i]])[0], self.classes)

            #Check that it's the right type
            self.assertIsInstance(model, LogisticRegression)
            #Check the model matrix has the right nummber of features
            self.assertEqual(model.coef_.shape[1], self.numFeatures)
            #Check that it has stored it in the right place
            self.assertTrue(os.path.exists(tmpdir + "/model/model.pkl"))
            #Check we can load it
            loadedModel = mlflow.sklearn.load_model(tmpdir + "/model")
            #Check that it's the right type
            self.assertIsInstance(loadedModel, LogisticRegression)
            #Check the model matrix matches the saved one
            self.assertEqual(model.coef_.shape, loadedModel.coef_.shape)

if __name__ == '__main__':
    unittest.main()