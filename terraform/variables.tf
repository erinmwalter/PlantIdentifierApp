variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "repository_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "plant-disease-app"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "plant-disease-app"
}

variable "image_name" {
  description = "Docker image name"
  type        = string
  default     = "plant-disease-app"
}