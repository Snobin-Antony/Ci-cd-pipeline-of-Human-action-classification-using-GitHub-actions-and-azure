import os
import json
import mlflow
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
ws = Workspace.from_config()

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help='Dataset path')
args = parser.parse_args()
mlflow.autolog()
mlflow.log_param("hello_param", "action_classifier")

data_csv=pd.read_csv(args.data)
# data_csv = pd.read_csv("human-activity-recognition-with-smartphones/human-activity-recognition-with-smartphones.csv")


print(data_csv.head())
print(data_csv["Activity"].unique())
print('Number of duplicates in data : ',sum(data_csv.duplicated()))
print('Total number of missing values in train : ', data_csv.isna().values.sum())

data_df = data_csv[~data_csv['Activity'].isin(['WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS'])]
print(data_df.Activity.value_counts())

# plt.figure(figsize=(10,8))
# plt.title('Barplot of Activity')
# sns.countplot(data_df.Activity)
# plt.xticks(rotation=0)
# plt.show()

# le = LabelEncoder()
# data_df['Activity'] = le.fit_transform(data_df.Activity)
# print(data_df['Activity'].sample(5))
# original_labels = le.inverse_transform([0,1,2,3])      # Only to know which one corresponds to each number

y_data_df = data_df.Activity
x_data_df = data_df.drop(['subject', 'Activity'], axis=1)
# x_data_df= x_data_df.iloc[:,0:2]

#Split the data and keep 20% back for testing later
X_train, X_test, Y_train, Y_test = train_test_split(x_data_df, y_data_df, test_size=0.20)
print("Train length", len(X_train))
print("Test length", len(X_test))

# # Create Decision Tree Classifier
# parameters = {'max_depth': np.arange(3, 15),'min_samples_split': np.arange(2, 11),'min_samples_leaf': np.arange(1, 11),'criterion': ['gini', 'entropy']}
# lr_classifier = DecisionTreeClassifier()
# Create Logistic Regression Classifier
parameters = {'C':np.arange(10,61,10), 'penalty':['l2','l1']}
lr_classifier = LogisticRegression()
lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters, cv=5,random_state = 42)
lr_classifier_rs.fit(X_train, Y_train)
y_pred = lr_classifier_rs.predict(X_test)

lr_accuracy = accuracy_score(y_true=Y_test, y_pred=y_pred)
print(f'Model - Accuracy: {lr_accuracy}')
mlflow.log_metric(f'Model_Accuracy:',lr_accuracy)

# Calculate AUC
Y_scores = lr_classifier_rs.predict_proba(X_test)
#print(Y_scores)
auc = roc_auc_score(Y_test, Y_scores, multi_class='ovr')
print(f'Model - AUC: {auc}')
mlflow.log_metric(f'Model_AUC:', auc)

## Make predictions
# output_class = lr_classifier_rs.predict(X_test.iloc[0:2])
output_class = lr_classifier_rs.predict(X_test.iloc[[1469]])
# output_class_label = original_labels[output_class]
# Convert the array to a list and then to JSON
activity_json = json.dumps(output_class.tolist())
# Print the predicted class
print(f"Predicted class: {activity_json}")

# function to plot confusion matrix
def plot_confusion_matrix(cm,lables):
    # labels  = original_labels[lables] 
    fig, ax = plt.subplots(figsize=(12,8)) # for plotting confusion matrix as image
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=lables, yticklabels=lables,
    ylabel='True label',
    xlabel='Predicted label')
    plt.xticks(rotation = 90)
    plt.title(f'Model Confusion Matrix\nAUC: {auc:.4f}')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# Plot the confusion matrix
cm = confusion_matrix(Y_test.values,y_pred)
plot_confusion_matrix(cm, np.unique(y_pred))  # plotting confusion matrix

registered_model_name="hac-model"

##########################
#<save and register model>
##########################
# Registering the model to the workspace
print("Registering the model via MLFlow")
mlflow.sklearn.log_model(
    sk_model=lr_classifier_rs,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name
)

# # Saving the model to a file
print("Saving the model via MLFlow")
mlflow.sklearn.save_model(
    sk_model=lr_classifier_rs,
    path=os.path.join(registered_model_name, "hac_model"),
)

# Replace with your workspace details
subscription_id = 'bf0717bf-dfd1-4019-a2b6-aa46e3899a4d'
resource_group = 'assignment-snobin'
workspace_name = 'assignmentsnobin'
service_name = 'hac-classifier-service'

# Load the Azure ML workspace
ws = Workspace.from_config()

# Retrieve the registered model
model = Model(ws, name=registered_model_name, version=1)  # Update version as needed

# Define the inference configuration
inference_config = InferenceConfig(entry_script='score.py', environment=Environment.get(ws, "coursework-ml-compute-human-action-classification"))

# Deploy the model as a web service using Azure Container Instances (ACI)
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1),
    overwrite=True  # Set to True if you want to overwrite an existing service with the same name
)

# Wait for the deployment to complete and display the service details
service.wait_for_deployment(show_output=True)
print(service.scoring_uri)
