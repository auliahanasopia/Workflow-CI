import warnings
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=False  
)

mlflow.set_experiment("CI-MLflow-DagsHub")

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True
)

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

with mlflow.start_run(run_name="rf-ci") as run:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy_manual", acc)

    cm = confusion_matrix(y_test, preds)

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
        artifact_path="model"
    )

    print(f"RUN_ID:{run.info.run_id}")



