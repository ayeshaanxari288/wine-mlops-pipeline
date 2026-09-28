import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data import load_and_split_data


def train_model():
    X_train, X_test, y_train, y_test = load_and_split_data()

    param_grid = [
        {"n_estimators": 10, "max_depth": 2},
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 15},
    ]

    mlflow.set_experiment("wine_classification")

    best_acc = 0.0
    best_run_id = None
    best_model = None

    for idx, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"run_config_{idx + 1}") as run:
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            if acc >= best_acc:
                best_acc = acc
                best_run_id = run.info.run_id
                best_model = model

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        model_version = mlflow.register_model(model_uri, "WineClassifier")
        
        client = MlflowClient()
        client.set_registered_model_alias(
            name="WineClassifier",
            alias="champion",
            version=model_version.version
        )
        print(f"Registered best model (Version {model_version.version}) with 'champion' alias!")

    return best_model, best_acc


if __name__ == "__main__":
    train_model()