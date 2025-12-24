import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("studentmat_preprocessing.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    registered_model_name="student-performance-model"
)




