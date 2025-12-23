import os
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

assert os.getenv("DAGSHUB_TOKEN"), "DAGSHUB_TOKEN is not set"
dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])

dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("CI-MLflow-DagsHub")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

mlflow.log_param("model", "RandomForest")
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")
