---
sidebar_position: 1
---

# vLLM Semantic Router

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/vllm-project/semantic-router/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Community-yellow)](https://huggingface.co/LLM-Semantic-Router)
[![Go Report Card](https://goreportcard.com/badge/github.com/vllm-project/semantic-router/src/semantic-router)](https://goreportcard.com/report/github.com/vllm-project/semantic-router/src/semantic-router)
![Test And Build](https://github.com/vllm-project/semantic-router/workflows/Test%20And%20Build/badge.svg)

An intelligent **Mixture-of-Models (MoM)** router that acts as an Envoy External Processor (ExtProc) to intelligently direct OpenAI API requests to the most suitable backend model from a defined pool. Using BERT-based semantic understanding and classification, it optimizes both performance and cost efficiency.

## 🚀 Key Features

### 🎯 **Auto-selection of Models**

Intelligently routes requests to specialized models based on semantic understanding:

- **Math queries** → Math-specialized models
- **Creative writing** → Creative-specialized models
- **Code generation** → Code-specialized models
- **General queries** → Balanced general-purpose models

### 🛡️ **Security & Privacy**

- **PII Detection**: Automatically detects and handles personally identifiable information
- **Prompt Guard**: Identifies and blocks jailbreak attempts
- **Safe Routing**: Ensures sensitive prompts are handled appropriately

### ⚡ **Performance Optimization**

- **Semantic Cache**: Caches semantic representations to reduce latency
- **Tool Selection**: Auto-selects relevant tools to reduce token usage and improve tool selection accuracy

### 🏗️ **Architecture**

- **Envoy ExtProc Integration**: Seamlessly integrates with Envoy proxy
- **Dual Implementation**: Available in both Go (with Rust FFI) and Python
- **Scalable Design**: Production-ready with comprehensive monitoring

## 📊 Performance Benefits

Our testing shows significant improvements in model accuracy through specialized routing:

![Model Accuracy](/img/category_accuracies.png)

## 🛠️ Architecture Overview

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="Architecture Overview" defaultZoom={3.1}>
{`graph TB
    Client[Client Request] --> Envoy[Envoy Proxy]
    Envoy --> Router[Semantic Router ExtProc]
    
    subgraph "Classification Modules"
        direction LR
        PII[PII Detector] 
        Jailbreak[Jailbreak Guard]
        Category[Category Classifier]
        Cache[Semantic Cache]
    end
    
    Router --> PII
    Router --> Jailbreak  
    Router --> Category
    Router --> Cache
    
    PII --> Decision{Security Check}
    Jailbreak --> Decision
    Decision -->|Block| Block[Block Request]
    Decision -->|Pass| Category
    Category --> Models[Route to Specialized Model]
    Cache -->|Hit| FastResponse[Return Cached Response]
    
    Models --> Math[Math Model]
    Models --> Creative[Creative Model] 
    Models --> Code[Code Model]
    Models --> General[General Model]`}
</ZoomableMermaid>

## 🎯 Use Cases

- **Enterprise API Gateways**: Route different types of queries to cost-optimized models
- **Multi-tenant Platforms**: Provide specialized routing for different customer needs
- **Development Environments**: Balance cost and performance for different workloads
- **Production Services**: Ensure optimal model selection with built-in safety measures

## 📈 Monitoring & Observability

The router provides comprehensive monitoring through:

- **Grafana Dashboard**: Real-time metrics and performance tracking
- **Prometheus Metrics**: Detailed routing statistics and performance data
- **Request Tracing**: Full visibility into routing decisions and performance

![LLM Router Dashboard](/img/grafana_screenshot.png)

## 🔗 Quick Links

- [**Installation**](installation/installation.md) - Setup and installation guide
- [**Overview**](overview/semantic-router-overview.md) - Deep dive into semantic routing concepts
- [**Architecture**](overview/architecture/system-architecture.md) - Technical architecture and design
- [**Model Training**](training/training-overview.md) - How classification models are trained
- [**Dashboard**](overview/dashboard.md) - Unified UI for config, monitoring, topology, and playground

## 📚 Documentation Structure

This documentation is organized into the following sections:

### 🎯 [Overview](overview/semantic-router-overview.md)

Learn about semantic routing concepts, mixture of models, and how this compares to other routing approaches like RouteLLM and GPT-5's router architecture.

### 🏗️ [Architecture](overview/architecture/system-architecture.md)

Understand the system design, Envoy ExtProc integration, and how the router communicates with backend models.

### 🤖 [Model Training](training/training-overview.md)

Explore how classification models are trained, what datasets are used, and the purpose of each model type.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/vllm-project/semantic-router/blob/main/LICENSE) file for details.
