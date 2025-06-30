from google.cloud import aiplatform
import argparse


def run_vertex_training_job(project_id, bucket_name, region="us-central1", job_name="patrain-job"):
    """Create and run a Vertex AI custom training job."""
    aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{bucket_name}")

    job = aiplatform.CustomJob(
        display_name=job_name,
        project=project_id,
        location=region,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
                "package_uris": [f"gs://{bucket_name}/patrain-0.1.0.tar.gz"],
                "python_module": "patrain.main",
            },
        }],
    )

    job.run(sync=True)
    print(f"Training job {job_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Vertex AI training job for patrain.")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--bucket", required=True, help="Cloud Storage bucket name")
    parser.add_argument("--region", default="us-central1", help="Vertex AI region")
    parser.add_argument("--job-name", default="patrain-job", help="Vertex AI job name")
    args = parser.parse_args()

    run_vertex_training_job(args.project_id, args.bucket, args.region, args.job_name)