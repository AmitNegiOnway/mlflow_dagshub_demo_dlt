import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='amitnegionway', repo_name='mlflow_dagshub_demo_dlt', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/amitnegionway/mlflow_dagshub_demo_dlt.mlflow") # if we worked on aws so we put here aws url.. 

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
max_depth = 5

n_estimator=100
# apply mlflow 

# mlflow.set_experiment('iris-dt') # this is of the option for experiment write a experiment name but also do experiment_id='479759578844802238'but for this u manually create new experiment in mlflow tool and then copy their experiment_id but in option 1 u dont need to do this.



mlflow.set_experiment('iris-rf')




with mlflow.start_run(): # this is called contaxt manager
    rf=RandomForestClassifier(max_depth=max_depth,n_estimator=n_estimator)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimator',n_estimator)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    # save the plot as an artifact
    plt.savefig('confusion_matrix.png')

    # mlflow artifact 
    mlflow.log_artifact('confusion_matrix.png')

    # mlflow code (hum code ko bhi log kar skte hai par normally dvc ye kar deta hai par mlflow mai ye ho jata hai.)
    mlflow.log_artifact(__file__) # here code is also a artifact..

    # how can log the model (means - this decision tree model.)
    mlflow.sklearn.log_model(rf,"random forest")

    # mlflow.log_model(dt,"decision tree") this one also used but if u used previous one they provided more meta data...
    
    # mlflow tags
    mlflow.set_tag('author','amit negi')
    mlflow.set_tag('model','random forest')
    mlflow.set_tag('day','6/26/2025')

    print('accuracy',accuracy)

