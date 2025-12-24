import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])
dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("CI-MLflow-Project")

with mlflow.start_run():
    data = pd.read_csv("studentmat_preprocessing.csv")
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
