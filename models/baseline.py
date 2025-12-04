import boto3
import pandas as pd
import os
import sagemaker
from sagemaker.automl.automl import AutoML
from sagemaker.inputs import TrainingInput
import mlflow
import json
import time
from sagemaker.exceptions import UnexpectedStatusException

# --- S3 Configuration ---
BUCKET = "nyc-taxi-ml-ops"
# Autopilot needs the *entire* dataset in one file for the initial step
INPUT_DATA_KEY = "data/processed/full_dataset.csv" 
# Output location for Autopilot to store its generated assets (e.g., candidates)
AUTOML_OUTPUT_PREFIX = "automl/autopilot-output" 
TARGET_COLUMN = "fare_amount"

# --- SageMaker/MLflow Configuration (INPUT YOUR OWN VALUES) ---
REGION = "us-east-2"
# Replace with your SageMaker Managed MLflow App Server Name
MLFLOW_SERVER_NAME = "GPQTPJPRSZ64" 
# Replace with your SageMaker execution role ARN (or use the default role if running in a notebook)
# If running locally, you must provide this.
# You can find the role ARN in the SageMaker console or by calling sagemaker.get_execution_role()
# if running inside a notebook instance.
# EXECUTION_ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-20230101T000000" 
# Set to 'None' or comment out if using a notebook role.
EXECUTION_ROLE_ARN = None 


def get_sagemaker_session_and_role():
    """Gets the SageMaker session and execution role."""
    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Use the role passed in, or try to infer from the session/environment
    role = EXECUTION_ROLE_ARN or sagemaker.get_execution_role(sagemaker_session)
    
    print(f"SageMaker Role: {role}")
    print(f"SageMaker Session Region: {sagemaker_session.boto_region_name}")
    return sagemaker_session, role


def setup_mlflow():
    """Configures the MLflow tracking URI for SageMaker Managed MLflow."""
    # This environment variable is CRITICAL for SigV4 authentication
    os.environ["MLFLOW_SAGEMAKER_AUTHORIZATION_REGION"] = REGION

    # The public DNS endpoint structure for the SageMaker Managed MLflow App.
    tracking_server_uri = f"https://app-{MLFLOW_SERVER_NAME}.mlflow.sagemaker.{REGION}.app.aws"

    # 1. Set the Tracking URI
    mlflow.set_tracking_uri(tracking_server_uri)

    # 2. Set the Experiment Name (Autopilot runs will appear here)
    EXPERIMENT_NAME = "NYC-Taxi-Fare-Prediction-Autopilot-Run"
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
    print(f"MLflow Experiment set to: {EXPERIMENT_NAME}")
    print("-" * 50)


def start_autopilot_job(sagemaker_session, role):
    """Starts a SageMaker Autopilot training job."""
    
    job_name = f'Autopilot-NYC-Fare-Reg-{int(time.time())}'
    
    # S3 input location of the full dataset
    input_data_uri = f's3://{BUCKET}/{INPUT_DATA_KEY}'
    
    # S3 output location for all generated assets
    output_location = f's3://{BUCKET}/{AUTOML_OUTPUT_PREFIX}/{job_name}'
    
    # 1. Define the Training Input (assumes the data is already uploaded as a single file)
    input_data = TrainingInput(
        s3_data=input_data_uri,
        content_type='text/csv',
        s3_data_type='S3Prefix'
    )

    # 2. Configure the Autopilot Estimator
    # Note: Autopilot automatically infers the problem type (regression in this case)
    autopilot_estimator = AutoML(
        role=role,
        target_attribute_name=TARGET_COLUMN,
        sagemaker_session=sagemaker_session,
        # Autopilot's objective metric for regression is typically MSE or RMSE, 
        # but we use 'RMSE' for display and it will use a related loss for training.
        # The default for regression is 'MSE'. Let's keep the default.
        # objective_metric='MSE',
        output_path=output_location,
        # 'ENSEMBLING' mode is the default and best for production
        max_candidates=10, # A small number for a quick run
        # Use a subset of data for fast testing if needed:
        # job_config={'DataSplitConfig': {'ValidationFraction': 0.2}},
        # Autopilot automatically logs its candidates, hyperparameters, and results 
        # to the active SageMaker Managed MLflow Experiment.
    )

    print(f"🚀 Starting SageMaker Autopilot Job: {job_name}")
    print(f"Input: {input_data_uri}")
    print(f"Output: {output_location}")

    # 3. Launch the job - use wait=False for non-blocking execution
    autopilot_estimator.fit(
        inputs=[input_data],
        job_name=job_name,
        wait=False
    )
    
    # 4. Return the estimator object to allow monitoring later
    return autopilot_estimator, job_name


def monitor_job_and_log_best(autopilot_estimator, job_name):
    """Polls the job status and logs the best candidate's metrics."""
    print("\n⏳ Waiting for Autopilot Job to complete...")
    
    try:
        # Blocking call to wait for the job to finish
        autopilot_estimator.wait()
        
        # Get the best candidate model
        best_candidate = autopilot_estimator.describe_auto_ml_job()['BestCandidate']
        best_candidate_name = best_candidate['CandidateName']
        best_candidate_status = best_candidate['CandidateStatus']
        
        # The final objective metric value
        final_metric_value = best_candidate['FinalAutoMLJobObjectiveMetric']['Value']

        print("\n✅ Autopilot Job Completed Successfully!")
        print(f"🏆 Best Candidate: {best_candidate_name} ({best_candidate_status})")
        print(f"🎯 Final Objective Metric (MSE): {final_metric_value:.4f}")
        
        # --- MLflow Logging of Final Result ---
        # While Autopilot already logs candidates, we can log the final best result 
        # to the top-level experiment run for easy visibility.
        with mlflow.start_run(run_name=f"{job_name}_Final_Summary") as run:
            mlflow.log_param("autopilot_job_name", job_name)
            mlflow.log_param("best_candidate_name", best_candidate_name)
            # Log the Autopilot's best metric.
            mlflow.log_metric("final_best_candidate_mse", final_metric_value)
            mlflow.log_param("autopilot_objective_metric", "MSE")
            
            # Log the model artifact URI for easy access
            model_artifact_uri = sagemaker.Model(
                model_data=best_candidate['CandidateProperties']['ModelInsights']['ModelArtifacts']['CandidateArtifacts'][0]['ArtifactDetails']['S3Uri'],
                role=autopilot_estimator.role,
                sagemaker_session=autopilot_estimator.sagemaker_session
            )
            # Log the model artifact path (S3 URI) to MLflow
            mlflow.log_param("best_model_s3_uri", model_artifact_uri.model_data)


    except UnexpectedStatusException as e:
        print(f"\n❌ Autopilot Job failed with status: {e}")
        # Log failure to MLflow
        with mlflow.start_run(run_name=f"{job_name}_Failed_Summary") as run:
            mlflow.log_param("autopilot_job_name", job_name)
            mlflow.log_param("status", "FAILED")
            mlflow.log_param("error_message", str(e))
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        

def main():
    """Main function to orchestrate the SageMaker Autopilot and MLflow process."""
    
    # 1. Setup SageMaker session and role
    sagemaker_session, role = get_sagemaker_session_and_role()
    
    # 2. Setup MLflow Tracking
    setup_mlflow()
    
    # 3. Start the Autopilot Job
    # NOTE: Ensure you have uploaded a combined 'full_dataset.csv' to s3://{BUCKET}/{INPUT_DATA_KEY}
    autopilot_estimator, job_name = start_autopilot_job(sagemaker_session, role)
    
    # 4. Monitor and Log Best Candidate
    # This call blocks until the Autopilot job is done.
    monitor_job_and_log_best(autopilot_estimator, job_name)

if __name__ == "__main__":
    main()