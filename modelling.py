import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Pastikan token ada (WAJIB untuk CI)
assert os.getenv("DAGSHUB_TOKEN"), "DAGSHUB_TOKEN is not set"
dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])

dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("CI-MLflow-Project")

data = pd.read_csv("studentmat_preprocessing.csv")

TARGET = "target"
X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )
