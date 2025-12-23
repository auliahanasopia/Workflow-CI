import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("CI-MLflow-DagsHub")

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True
    )

    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset train_pca.csv tidak ditemukan")

    data = pd.read_csv(file_path)

    X = data.drop("Credit_Score", axis=1)
    y = data["Credit_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    input_example = X_train.head(5)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    with mlflow.start_run(run_name="rf-ci-advanced") as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Manual metric tambahan (aman walau autolog aktif)
        mlflow.log_metric("accuracy_manual", accuracy)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"RUN_ID:{run.info.run_id}")





