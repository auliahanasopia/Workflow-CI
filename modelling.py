import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

assert os.getenv("DAGSHUB_TOKEN"), "DAGSHUB_TOKEN is not set"

dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("CI-MLflow-Project")

DATA_PATH = "studentmat_preprocessing.csv"
TARGET = "target"

data = pd.read_csv(DATA_PATH)

X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="rf-ci", nested=True) as run:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"RUN_ID_TAG:{run.info.run_id}")



