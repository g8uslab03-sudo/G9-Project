# GCP Cloud Run Service Definitions
# Deploy each service with: gcloud run deploy SERVICE_NAME --source . --region us-central1

# api-gateway
gcloud run deploy multi-agent-api-gateway \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 80 \
  --timeout 300 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=api-gateway" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# orchestrator
gcloud run deploy multi-agent-orchestrator \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8081 \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 10 \
  --timeout 600 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=orchestrator" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# memory-service
gcloud run deploy multi-agent-memory \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8082 \
  --cpu 1 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 50 \
  --timeout 300 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=memory-service,HOT_TIER_GB=2,WARM_TIER_GB=8,COLD_TIER_GB=50,VECTOR_COMPRESSION=RaBitQ" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# rag-service
gcloud run deploy multi-agent-rag \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8083 \
  --cpu 1 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 20 \
  --timeout 300 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=rag-service,EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2,VECTOR_INDEX_TYPE=HNSW,TOP_K=10" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# skill-service
gcloud run deploy multi-agent-skills \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8084 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 30 \
  --timeout 300 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=skill-service,TOTAL_SKILLS=250,CATEGORIES=5" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# research-service
gcloud run deploy multi-agent-research \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8085 \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 5 \
  --timeout 3600 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=research-service,SANDBOX_TIMEOUT=3600,MAX_EXPERIMENTS=10" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# dream-service (scheduled via Cloud Scheduler)
gcloud run deploy multi-agent-dream \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8086 \
  --cpu 0.5 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 1 \
  --timeout 3600 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=dream-service,SCHEDULE=0 2 * * *,HYPOTHESIS_COUNT=5" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# magpie-service (scheduled via Cloud Scheduler)
gcloud run deploy multi-agent-magpie \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8087 \
  --cpu 0.5 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 1 \
  --timeout 3600 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=magpie-service,SCHEDULE=0 3 * * *,TRAINING_PRECISION=bf16" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# monitoring
gcloud run deploy multi-agent-monitoring \
  --source . \
  --region us-central1 \
  --platform managed \
  --no-allow-unauthenticated \
  --port 9090 \
  --cpu 0.25 \
  --memory 512Mi \
  --min-instances 1 \
  --max-instances 1 \
  --concurrency 10 \
  --timeout 300 \
  --set-env-vars "ENV=production,LOG_LEVEL=INFO,SERVICE_NAME=monitoring" \
  --service-account multi-agent-sa@my-project.iam.gserviceaccount.com

# Create Cloud Scheduler jobs for dream and magpie modules
gcloud scheduler jobs create http multi-agent-dream-daily \
  --schedule="0 2 * * *" \
  --uri="https://multi-agent-dream-xyz.run.app/trigger" \
  --http-method=POST \
  --location=us-central1 \
  --time-zone="UTC" \
  --description="Daily dream cycle for hypothesis generation"

gcloud scheduler jobs create http multi-agent-magpie-weekly \
  --schedule="0 3 * * 0" \
  --uri="https://multi-agent-magpie-xyz.run.app/trigger" \
  --http-method=POST \
  --location=us-central1 \
  --time-zone="UTC" \
  --description="Weekly Magpie training cycle"