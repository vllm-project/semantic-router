import React from 'react';
import Layout from '@theme/Layout';
import styles from './roadmap.module.css';

const priorityColors = {
  'P0': '#dc3545', // Red for critical
  'P1': '#fd7e14', // Orange for important
  'P2': '#6c757d', // Gray for nice-to-have
};

const PriorityBadge = ({ priority }) => (
  <span 
    className={styles.priorityBadge}
    style={{ backgroundColor: priorityColors[priority] }}
  >
    {priority}
  </span>
);

const RoadmapItem = ({ title, priority, acceptance, children }) => (
  <div className={styles.roadmapItem}>
    <div className={styles.itemHeader}>
      <h4 className={styles.itemTitle}>{title}</h4>
      <PriorityBadge priority={priority} />
    </div>
    {children && <div className={styles.itemDescription}>{children}</div>}
    {acceptance && (
      <div className={styles.acceptance}>
        <strong>Acceptance:</strong> {acceptance}
      </div>
    )}
  </div>
);

const AreaSection = ({ title, children }) => (
  <div className={styles.areaSection}>
    <h3 className={styles.areaTitle}>{title}</h3>
    <div className={styles.areaContent}>
      {children}
    </div>
  </div>
);

export default function RoadmapV01() {
  return (
    <Layout
      title="Roadmap v0.1"
      description="vLLM Semantic Router v0.1 Development Roadmap"
    >
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <h1>Roadmap v0.1</h1>
            <p className={styles.subtitle}>
              Productizing Intelligent Routing with Comprehensive Evaluation
            </p>
            
            <div className={styles.goalSection}>
              <h2>Release Goal</h2>
              <p>
                This release focuses on productizing the semantic router with:
              </p>
              <ol>
                <li>Intelligent routing with configurable reasoning modes and model-family-aware templating</li>
                <li>Kubernetes-native deployment with auto-configuration from model evaluation</li>
                <li>Comprehensive benchmarking and monitoring beyond MMLU-Pro</li>
                <li>Production-ready caching and observability</li>
              </ol>
              
              <div className={styles.keyDeliverables}>
                <h3>Key P0 Deliverables</h3>
                <ul>
                  <li><strong>Router intelligence:</strong> Reasoning controller, ExtProc plugins, semantic caching</li>
                  <li><strong>Operations:</strong> K8s operator, benchmarks, monitoring</li>
                  <li><strong>Quality:</strong> Test coverage, integration tests, structured logging</li>
                </ul>
              </div>
            </div>

            <div className={styles.priorityLegend}>
              <h3>Priority Criteria</h3>
              <div className={styles.priorityItems}>
                <div className={styles.priorityItem}>
                  <PriorityBadge priority="P0" />
                  <div>
                    <strong>Critical / Must-Have</strong>
                    <p>Directly impacts core functionality or correctness. Without this, the system cannot be reliably used in production.</p>
                  </div>
                </div>
                <div className={styles.priorityItem}>
                  <PriorityBadge priority="P1" />
                  <div>
                    <strong>Important / Should-Have</strong>
                    <p>Improves system quality, efficiency, or usability but is not blocking the basic workflow.</p>
                  </div>
                </div>
                <div className={styles.priorityItem}>
                  <PriorityBadge priority="P2" />
                  <div>
                    <strong>Nice-to-Have / Exploratory</strong>
                    <p>Experimental or advanced features that extend system capability.</p>
                  </div>
                </div>
              </div>
            </div>

            <AreaSection title="RouterCore (area/core)">
              <div className={styles.subsection}>
                <h4>Model Selection and Configuration</h4>
                <RoadmapItem
                  title="Reasoning mode controller"
                  priority="P0"
                  acceptance="Configurable reasoning effort levels per category; template handling for different model families (GPT OSS/Qwen3/DeepSeek/etc); metrics for reasoning mode decisions and model-specific template usage."
                />
              </div>
              
              <div className={styles.subsection}>
                <h4>Routing Logic</h4>
                <RoadmapItem
                  title="ExtProc modular architecture"
                  priority="P0"
                  acceptance="Interface for different classifiers and functionality modules; config for feature gating and module enablement; Plugin API versioning."
                />
                <RoadmapItem
                  title="Load and latency-aware endpoint resilience"
                  priority="P2"
                  acceptance="Endpoint selection using request concurrency and/or TTFT/TPOT. Using SLO driven metrics to automatic failover with load weighted selection between redundant endpoints; circuit breaker with error rate and load signal deviation thresholds."
                />
              </div>
              
              <div className={styles.subsection}>
                <h4>Semantic Cache</h4>
                <RoadmapItem
                  title="Production-ready semantic caching"
                  priority="P0"
                  acceptance="Support more backends in semantic caching; hit rates are tracked; cache eviction is configurable; performance benchmarks are included."
                />
              </div>
            </AreaSection>

            <AreaSection title="Research (area/research)">
              <RoadmapItem
                title="Multi-factor routing algorithm"
                priority="P1"
                acceptance="Routing formula combining quality (model_scores), load (ModelLoad counter), and latency (ModelCompletionLatency histogram), and token usage and pricing; configurable for broad SLO based targets; documented in architecture guide."
              />
                <RoadmapItem
                  title="Dynamic model scoring system"
                  priority="P2"
                  acceptance="Online model score updates based on model accuracy, latency, and cost metrics; auto-updates model_scores in config; replaces static scoring in A/B test or through RL."
                />
            </AreaSection>

            <AreaSection title="Networking (area/networking)">
              <RoadmapItem
                title="Envoy ExtProc integration for AI gateways"
                priority="P0"
                acceptance="1. ExtProc header/body mutation consistent with LLM-d/Envoy AI Gateway filter chains (documented setup for each) 2. Example Envoy configs for common patterns (e.g., A/B test, canary routing)"
              />
            </AreaSection>

            <AreaSection title="Bench (area/benchmark)">
              <RoadmapItem
                title="Router benchmark CLI"
                priority="P0"
                acceptance="Command run_bench.sh with: 1. Per-category metrics: accuracy, response time, token counts (prompt/completion/total) 2. Per-model metrics: success rate, error distribution, latency distribution 3. Export to CSV/JSON for analysis"
              />
              <RoadmapItem
                title="Performance test suite"
                priority="P0"
                acceptance="Comprehensive test framework (including but not limited to MMLU-Pro accuracy, PII/jailbreak detection, latency) with configurable thresholds; CI integration with baseline metrics."
              />
              <RoadmapItem
                title="Reasoning mode evaluation"
                priority="P1"
                acceptance="Compare standard vs. reasoning mode using: 1. Response correctness on MMLU(-Pro) and non-MMLU test sets 2. Token usage (completion_tokens/prompt_tokens ratio) 3. Response time per output token"
              />
            </AreaSection>

            <AreaSection title="User Experience (area/user-experience)">
              <RoadmapItem
                title="Developer quickstart examples"
                priority="P0"
                acceptance="A new user can reproduce an evaluation report in under 10 minutes."
              />
            </AreaSection>

            <AreaSection title="Test and Release (area/tooling, area/ci)">
              <RoadmapItem
                title="More ExtProc test coverage"
                priority="P0"
                acceptance="1. Ginkgo test suite for ExtProc components (request/response handling, model selection) 2. Integration tests for model-specific reasoning (GPT OSS/Qwen3/DeepSeek template kwargs) 3. Config validation tests (model paths, thresholds, cache settings)"
              />
              <RoadmapItem
                title="Classifier test framework"
                priority="P1"
                acceptance="1. Test vectors for category/PII/jailbreak detection with expected outputs 2. Mock BERT model for fast CI runs 3. Snapshot tests for classification boundaries"
              />
            </AreaSection>

            <AreaSection title="Environment (area/environment)">
              <RoadmapItem
                title="Container and k8s deployment readiness"
                priority="P0"
                acceptance="K8s manifests with model init container, health/readiness probes, resource limits, and metrics endpoints; documented deployment flow. Operator that uses LLM model eval to generate config yaml and startup the k8s deployment"
              />
              <RoadmapItem
                title="Model management automation"
                priority="P1"
                acceptance="Automated model download/verification from HuggingFace; version pinning; graceful fallback on missing models or revisions; HuggingFace model uploading CI to ensure models are fully evaluated before overwriting existing models."
              />
            </AreaSection>

            <AreaSection title="Observability (area/observability)">
              <RoadmapItem
                title="Minimal operator dashboard"
                priority="P0"
                acceptance="Grafana panels from logs/metrics for reasoning rate, cost, latency, and refusal rates."
              />
              <RoadmapItem
                title="Structured logs and metrics"
                priority="P0"
                acceptance="Model choice, reasoning flag, token counts, cost, and reason codes are emitted; alerts are configurable."
              />
              <RoadmapItem
                title="Routing policy visualization"
                priority="P1"
                acceptance="Grafana dashboard showing routing flow (source->target models), confidence distributions, and cost metrics; alerts on threshold violations."
              />
            </AreaSection>

            <AreaSection title="Docs (area/document)">
              <RoadmapItem
                title="Reasoning routing quickstart"
                priority="P0"
                acceptance="Short guide with config.yaml fields, example request/response, and a comprehensive evaluation command, within a recorded video for demo the reasoning use case.."
              />
              <RoadmapItem
                title="Policy cookbook and troubleshooting"
                priority="P1"
                acceptance="Short recipes with config.yaml snippets for categories/model_scores & use_reasoning, classifier thresholds, model_config PII policy, and vllm_endpoints mapping; troubleshooting maps common logs/errors to exact config fixes."
              />
              <RoadmapItem
                title="Model performance evaluation guide"
                priority="P1"
                acceptance="Documents automated workflow to evaluate models (including but not limited to MMLU-Pro), generate performance-based routing config, and update categories[].model_scores; includes example evaluation->config pipeline."
              />
            </AreaSection>
          </div>
        </div>
      </div>
    </Layout>
  );
}
