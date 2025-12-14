---
title: Multimodal Routing
sidebar_label: Multimodal Routing
---

# Multimodal Intent Classification and Routing

This guide explains how to configure and use multimodal routing in the Semantic Router, enabling intelligent routing of requests containing both text and images.

## Overview

The Semantic Router supports multimodal intent classification using two complementary approaches:

1. **Embedding-Based Multimodal Classification**: For visual analysis and understanding tasks
   - Uses CLIP vision transformer for image embeddings
   - Fuses text (BERT/Qwen3) and image (CLIP) embeddings
   - Zero-shot classification via cosine similarity matching

2. **Image Generation Intent Detection**: For image generation intent detection
   - Fine-tuned BERT classifier
   - Distinguishes between image generation and non-image generation requests

### Key Components

- **CLIP Vision Transformer**: Extracts semantic image embeddings
- **BERT/Qwen3 Text Encoder**: Extracts semantic text embeddings
- **Embedding Fusion**: Weighted combination with L2 normalization
- **Category Descriptions**: Text descriptions used for similarity matching

## Configuration

### Enable Multimodal for a Category

```yaml
categories:
  - name: visual_analysis
    description: "Image analysis, object detection, visual reasoning, scene understanding"
    multimodal_enabled: true
    model_scores:
      - model: llava:7b
        score: 1.0
        capabilities: ["text", "image"]
```

**Required Fields:**
- `multimodal_enabled: true` - Enables embedding-based classification
- `description` - Used to generate category embeddings (must be descriptive)
- `capabilities: ["text", "image"]` - Specifies model capabilities

### Image Generation Category

```yaml
categories:
  - name: image_generation
    description: "Generate images, create pictures, draw illustrations"
    model_scores:
      - model: image-generator
        score: 1.0
        capabilities: ["image"]
```

## Usage

### OpenAI-Compatible API Format

The router supports OpenAI-compatible request formats:

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64_DATA"}}
      ]
    }]
  }'
```

### Direct Classification API

```bash
# Encode image
IMAGE_B64=$(base64 -i cat.jpg | tr -d '\n')

# Classify multimodal request
curl -X POST http://localhost:8080/api/v1/classify/multimodal \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"What is in this image?\",
    \"images\": [{
      \"data\": \"$IMAGE_B64\",
      \"mime_type\": \"image/jpeg\"
    }]
  }"
```

### Response Format

```json
{
  "classification": {
    "category": "visual_analysis",
    "confidence": 0.85,
    "processing_time_ms": 523
  },
  "content_type": "multimodal",
  "recommended_model": "llava:7b",
  "routing_decision": "route"
}
```

## How It Works

### Embedding Extraction

1. **Text Embedding**: 
   - Tries Qwen3/Gemma (512-dim) first
   - Falls back to BERT (384-dim) if unavailable
   - Projects to 512 dimensions for fusion

2. **Image Embedding**:
   - CLIP vision transformer extracts 512-dim embeddings
   - Supports multiple images (averaged element-wise)
   - Preprocessing: decode, resize to 224x224, CLIP normalization

### Embedding Fusion

```go
// Weighted combination: 50% text + 50% image
fused = 0.5 * textEmbedding + 0.5 * imageEmbedding

// L2 normalization for cosine similarity
fused = fused / ||fused||
```

### Classification

- Computes cosine similarity between fused embedding and all category description embeddings
- Returns category with highest similarity score
- Confidence score is the similarity value (0.0-1.0)

## Performance Characteristics

### Latency Breakdown

- **Image Embedding Extraction**: ~200-300 ms (CLIP inference)
- **Fusion & Classification**: <1 second

### Optimization Strategies

**Batch Processing**: Future enhancement for multiple images

### Error Handling

The system includes automatic fallbacks:
- Falls back to Ollama/llava:7b if CLIP is unavailable
- Falls back to BERT if Qwen3/Gemma is unavailable
- Returns appropriate error responses for invalid inputs

## Troubleshooting

### Vision Transformer Not Available

If you see errors about vision transformer initialization:
- Check that CLIP model is downloaded
- Verify Rust/Candle bindings are compiled
- Check logs for initialization errors

## Examples

### Visual Analysis Request

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe what you see in this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

**Expected Routing**: `visual_analysis` → `llava:7b`

### Image Generation Request

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{
      "role": "user",
      "content": "Generate an image of a cat"
    }]
  }'
```

**Expected Routing**: `image_generation` → `image-generator`

## See Also

- [Categories Configuration](../cookbook/categories-configuration.md) - Detailed configuration guide
- [API Reference](../api/classification.md) - Complete API documentation
- [Embedding Routing](./embedding-routing.md) - Text-only embedding-based routing

