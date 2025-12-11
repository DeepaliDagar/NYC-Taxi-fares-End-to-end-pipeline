import os
import sagemaker
from sagemaker.automl.automl import AutoML
from automl import automl as training_steps
from automl import track_experiments

MODEL_PACKAGE_GROUP = os.environ.get("SM_MODEL_PACKAGE_GROUP_NAME")

if not MODEL_PACKAGE_GROUP:
    raise EnvironmentError(
        "The environment variable SM_MODEL_PACKAGE_GROUP_NAME must be set."
    )

def main():
    print("\n" + "="*50)
    print("PIPELINE STEP 1: Starting AutoML Training")
    print("="*50)
    
    session, role = training_steps.get_sagemaker_session_and_role()
    estimator, job_name = training_steps.start_autopilot_job(session, role)
    best_candidate = training_steps.monitor_job_and_display_results(estimator, job_name)
    
    if not best_candidate:
        print("Pipeline stopped: AutoML job failed.")
        return
    
    print("\n" + "="*50)
    print("PIPELINE STEP 2: Logging to MLflow")
    print("="*50)

    track_experiments.run_tracking(job_name)
    
    print("\n" + "="*50)
    print("PIPELINE STEP 3: Registering to SageMaker Registry")
    print("="*50)
    
    try:
        auto_ml_job = AutoML.attach(auto_ml_job_name=job_name)
        best_cand = auto_ml_job.best_candidate()
        
        model = auto_ml_job.create_model(
            name=best_cand['CandidateName'],
            candidate=best_cand,
            inference_response_keys=None 
        )
        
        model.register(
            content_types=['text/csv'],
            response_types=['text/csv'],
            inference_instances=['ml.m5.large'],
            transform_instances=['ml.m5.large'],
            model_package_group_name=MODEL_PACKAGE_GROUP, 
            approval_status='PendingManualApproval'
        )
        print(f"Successfully registered model: {best_cand['CandidateName']}")
        print(f"Go to SageMaker Console -> Model Registry to approve deployment.")
        
        
    except Exception as e:
        print(f"Failed to register in SageMaker: {e}")

if __name__ == "__main__":
    main()