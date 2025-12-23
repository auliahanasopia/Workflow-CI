import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

mlflow.set_experiment("CI-MLflow-Project")

data = pd.read_csv("studentmat_preprocessing.csv")
X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=200, max_depth=20)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print(f"RUN_ID_TAG:{mlflow.active_run().info.run_id}")
