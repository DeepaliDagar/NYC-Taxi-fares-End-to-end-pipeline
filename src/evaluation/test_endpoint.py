import boto3
import pandas as pd
import numpy as np
import math
import io
import os
from sklearn.metrics import mean_squared_error


ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
FILE_KEY = os.environ.get("S3_TEST_DATA_KEY", "processed/test.csv") 
AWS_REGION = os.environ.get("AWS_REGION")

if not ENDPOINT_NAME or not BUCKET_NAME or not AWS_REGION:
    raise EnvironmentError(
        "The environment variables SAGEMAKER_ENDPOINT_NAME, S3_BUCKET_NAME, and AWS_REGION must be set."
    )


runtime_client = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION) 

print(f"Downloading {FILE_KEY} from {BUCKET_NAME}...")
obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

print(f"Loaded {len(df)} rows.")

target_col = 'fare_amount'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' missing.")

y_actual = df[target_col].values
X_features = df.drop(columns=[target_col])

predictions = []
print(f"Starting inference against endpoint: {ENDPOINT_NAME}...") 

for index, row in X_features.iterrows():
    payload = ",".join(map(str, row.values))
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=payload
        )
        
        result_str = response['Body'].read().decode('utf-8')
        
        pred_value = float(result_str.split(',')[0].strip('[]" \n'))
        predictions.append(pred_value)

        if index % 100 == 0:
            print(f"Row {index}: Actual={y_actual[index]:.2f}, Pred={pred_value:.2f}")

    except Exception as e:
        print(f"Error on row {index} - Endpoint communication failed: {e}")
        predictions.append(np.nan)


valid_mask = ~np.isnan(predictions)
y_clean = y_actual[valid_mask]
preds_clean = np.array(predictions)[valid_mask]

if len(preds_clean) > 0:
    mse = mean_squared_error(y_clean, preds_clean)
    rmse = math.sqrt(mse)
    
    print("\n" + "="*40)
    print(f"FINAL RMSE: {rmse:.4f}")
    print(f"Processed {len(preds_clean)}/{len(df)} records.")
    print("="*40)
else:
    print("No valid predictions made.")