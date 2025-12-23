import os
import dagshub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# DAGsHub init (MLflow tracking otomatis)
# =========================
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USER", "auliahanasopia"),
    repo_name="student-performance-mlops",
    mlflow=True
)

mlflow.set_experiment("student-performance-mlops")
mlflow.autolog()

# =========================
# Load data
# =========================
df = pd.read_csv("studentmat_preprocessing.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pastikan integer columns jadi float
int_cols = X_train.select_dtypes(include="int").columns
X_train[int_cols] = X_train[int_cols].astype("float64")
X_test[int_cols] = X_test[int_cols].astype("float64")

# =========================
# Train + log model ke DAGsHub MLflow
# =========================
model = LogisticRegression(max_iter=1000)

with mlflow.start_run():
    model.fit(X_train, y_train)

    # metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # log confusion matrix
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("images/confusion_matrix.png", artifact_path="images")

    # log & register model
    mlflow.sklearn.log_model(
        model,
        name="StudentPerformanceLogisticRegressionModel"
    )

print("Model, metrics, dan artifact berhasil dicatat di DAGsHub MLflow")
