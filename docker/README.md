# Docker Compose Quick Start Guide

This Docker Compose configuration allows you to quickly run Semantic Router + Envoy proxy locally.

## Prerequisites

- Docker and Docker Compose
- Ensure ports 8801, 50051, 19000 are not in use

## Install in Docker Compose

1. **Clone the repository and navigate to the project directory**

   ```bash
   git clone https://github.com/vllm-project/semantic-router.git
   cd semantic_router
   ```

2. **Download required models** (if not already present):

   ```bash
   make download-models
   ```

   This will download the necessary ML models for classification:
   - Category classifier (ModernBERT-base)
   - PII classifier (ModernBERT-base)
   - Jailbreak classifier (ModernBERT-base)

3. **Start the services using Docker Compose**

   ```bash
   # Start core services (semantic-router + envoy)
   docker-compose up --build

   # Or run in background
   docker-compose up --build -d

   # Start with testing services (includes mock vLLM)
   docker-compose --profile testing up --build
   ```

4. **Verify the installation**
   - Semantic Router: http://localhost:50051 (gRPC service)
   - Envoy Proxy: http://localhost:8801 (main endpoint)
   - Envoy Admin: http://localhost:19000 (admin interface)

## Quick Start

### 1. Build and Start Services

```bash
# Start core services (semantic-router + envoy)
docker-compose up --build

# Or run in background
docker-compose up --build -d
```
