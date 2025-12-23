import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load data
data = pd.read_csv("studentmat_preprocessing.csv")

TARGET = "G3"

X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_example = X_train.iloc[:5]

mlflow.set_experiment("CI-MLflow-Project")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print(f"RUN_ID_TAG:{mlflow.active_run().info.run_id}")


