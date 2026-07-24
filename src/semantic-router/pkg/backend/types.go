package backend

import (
	"strings"
	"time"
)

// EngineKind identifies the serving engine that produced backend telemetry.
type EngineKind string

const (
	EngineKindVLLM        EngineKind = "vllm"
	EngineKindSGLang      EngineKind = "sglang"
	EngineKindATOM        EngineKind = "atom"
	EngineKindTensorRTLLM EngineKind = "tensorrt-llm"
)

// HealthState reports the adapter's normalized view of backend health.
type HealthState string

const (
	HealthStateUnknown   HealthState = "unknown"
	HealthStateHealthy   HealthState = "healthy"
	HealthStateDegraded  HealthState = "degraded"
	HealthStateUnhealthy HealthState = "unhealthy"
)

// BackendIdentity names one logical backend and optional serving replica.
type BackendIdentity struct {
	BackendID     string
	ModelName     string
	ReplicaID     string
	Endpoint      string
	EngineKind    EngineKind
	EngineVersion string
}

// Normalize trims string identity fields while preserving engine metadata.
func (id BackendIdentity) Normalize() BackendIdentity {
	id.BackendID = strings.TrimSpace(id.BackendID)
	id.ModelName = strings.TrimSpace(id.ModelName)
	id.ReplicaID = strings.TrimSpace(id.ReplicaID)
	id.Endpoint = strings.TrimSpace(id.Endpoint)
	id.EngineVersion = strings.TrimSpace(id.EngineVersion)
	return id
}

// Key returns the stable store key for a model/backend/replica tuple.
func (id BackendIdentity) Key() string {
	id = id.Normalize()
	if id.ReplicaID == "" {
		return id.ModelName + "|" + id.BackendID
	}
	return id.ModelName + "|" + id.BackendID + "|" + id.ReplicaID
}

// BackendTelemetry is the engine-neutral telemetry contract consumed by
// backend-aware policy helpers.
type BackendTelemetry struct {
	Identity BackendIdentity

	QueueDepth     *int
	ActiveRequests *int

	GPUUtilization  *float64
	MemoryPressure  *float64
	KVCachePressure *float64

	Latency  BackendLatency
	Affinity BackendAffinity

	Health      HealthState
	Confidence  float64
	CollectedAt time.Time
	TTL         time.Duration
}

// BackendLatency holds normalized latency snapshots in seconds.
type BackendLatency struct {
	TTFTSeconds  LatencySnapshot
	TPOTSeconds  LatencySnapshot
	E2ESeconds   LatencySnapshot
	QueueSeconds LatencySnapshot
}

// LatencySnapshot reports latency distribution samples in seconds.
type LatencySnapshot struct {
	P50Seconds *float64
	P90Seconds *float64
	P95Seconds *float64
	P99Seconds *float64
}

// BackendAffinity carries optional cache/session hints without engine-specific
// policy logic.
type BackendAffinity struct {
	PrefixCacheHitRate *float64
	KVCacheReuseScore  *float64
	SessionAffinity    map[string]float64
	ExtraHints         map[string]float64
}

// BackendCandidate is the minimal input shape for second-stage backend policy.
type BackendCandidate struct {
	BackendID    string
	ReplicaID    string
	ModelName    string
	EndpointName string
	Weight       int
}

// BackendPolicyResult explains the selected backend or why policy failed open.
type BackendPolicyResult struct {
	SelectedBackendID string                   `json:"selected_backend_id,omitempty"`
	SelectedReplicaID string                   `json:"selected_replica_id,omitempty"`
	FailOpen          bool                     `json:"fail_open"`
	Reason            string                   `json:"reason,omitempty"`
	Diagnostics       BackendPolicyDiagnostics `json:"diagnostics"`
}

// BackendPolicyDiagnostics is safe to persist in replay/debug records.
type BackendPolicyDiagnostics struct {
	SelectedModel       string `json:"selected_model,omitempty"`
	SelectedBackendID   string `json:"selected_backend_id,omitempty"`
	SelectedReplicaID   string `json:"selected_replica_id,omitempty"`
	TelemetryFresh      bool   `json:"telemetry_fresh"`
	TelemetryAgeMS      int64  `json:"telemetry_age_ms,omitempty"`
	PolicyReason        string `json:"policy_reason,omitempty"`
	FallbackReason      string `json:"fallback_reason,omitempty"`
	CandidateCount      int    `json:"candidate_count"`
	FreshCandidateCount int    `json:"fresh_candidate_count"`
	UnhealthyCount      int    `json:"unhealthy_count"`
}
