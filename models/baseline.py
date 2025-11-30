import boto3
import pandas as pd
import io
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

BUCKET = "nyc-taxi-ml-ops"
TRAIN_KEY = "processed/train_final.csv"
TEST_KEY = "processed/test_final.csv"

OUTPUT_METRICS = "baseline/baseline_metrics.json"
OUTPUT_MODEL = "baseline/baseline_model.pkl"


def main():
    s3 = boto3.client("s3")

    print("Downloading processed train/test datasets from S3...")
    # Load train
    train_obj = s3.get_object(Bucket=BUCKET, Key=TRAIN_KEY)
    train_df = pd.read_csv(io.BytesIO(train_obj["Body"].read()))

    # Load test
    test_obj = s3.get_object(Bucket=BUCKET, Key=TEST_KEY)
    test_df = pd.read_csv(io.BytesIO(test_obj["Body"].read()))

    print("Train:", train_df.shape, " Test:", test_df.shape)

    # Separate features/targets
    target = "fare_amount"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # ---------------------------
    # BASELINE MODEL
    # ---------------------------
    print("Training baseline RandomForest model...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("Baseline RMSE:", rmse)
    print("Baseline R²:", r2)

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2)
    }

    # ---------------------------
    # UPLOAD BASELINE METRICS
    # ---------------------------
    print("Uploading baseline metrics to S3...")

    s3.put_object(
        Bucket=BUCKET,
        Key=OUTPUT_METRICS,
        Body=json.dumps(metrics, indent=4).encode("utf-8")
    )

    # ---------------------------
    # UPLOAD BASELINE MODEL (Pickle)
    # ---------------------------
    print("Uploading baseline model to S3...")

    pickle_buffer = io.BytesIO()
    pickle.dump(model, pickle_buffer)

    s3.put_object(
        Bucket=BUCKET,
        Key=OUTPUT_MODEL,
        Body=pickle_buffer.getvalue()
    )

    print("DONE!")
    print("Metrics stored at:", f"s3://{BUCKET}/{OUTPUT_METRICS}")
    print("Model stored at:", f"s3://{BUCKET}/{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
