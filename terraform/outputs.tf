output "cloud_run_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.plant_disease_app.uri
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository path"
  value       = google_artifact_registry_repository.plant_disease_app.id
}

output "docker_image_path" {
  description = "Full Docker image path"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/${var.image_name}"
}