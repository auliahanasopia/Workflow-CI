import mlflow
import mlflow.sklearn
import dagshub
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi DagsHub
dagshub.init(
    repo_owner="auliahanasopia",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("CI-MLflow-DagsHub")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gunakan nested=True agar tidak error saat dipanggil lewat 'mlflow run'
with mlflow.start_run(run_name="rf-ci", nested=True) as run:
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    mlflow.log_metric("accuracy", acc)
    # Log model ke DagsHub (ini akan menghasilkan model URI)
    mlflow.sklearn.log_model(model, "model")
    
    # Cetak Run ID agar bisa ditangkap oleh GitHub Actions
    print(f"RUN_ID_OUPUT:{run.info.run_id}")
