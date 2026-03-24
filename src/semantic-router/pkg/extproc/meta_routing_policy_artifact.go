package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

const (
	metaRoutingPolicyArtifactVersion    = "meta-routing-policy/v1alpha1"
	metaRoutingPolicyFeatureSchemaName  = "feedback_record_flattened"
	metaRoutingPolicyFeatureSchemaVer   = "v1"
	metaRoutingPolicyProviderCalibrated = "calibrated_policy"
	metaRoutingPolicyProviderLearned    = "learned_policy"
)

type metaRoutingPolicyArtifact struct {
	Version       string                          `json:"version"`
	ArtifactID    string                          `json:"artifact_id,omitempty"`
	Provider      metaRoutingPolicyArtifactSource `json:"provider"`
	FeatureSchema metaRoutingPolicyFeatureSchema  `json:"feature_schema"`
	Rollout       metaRoutingPolicyRolloutGate    `json:"rollout"`
	Evaluation    metaRoutingPolicyEvaluation     `json:"evaluation"`
	Policy        metaRoutingPolicyBody           `json:"policy"`
}

type metaRoutingPolicyArtifactSource struct {
	Kind    string `json:"kind"`
	Name    string `json:"name"`
	Version string `json:"version"`
}

type metaRoutingPolicyFeatureSchema struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type metaRoutingPolicyRolloutGate struct {
	MinReplayRecords     int      `json:"min_replay_records,omitempty"`
	MinTriggerPrecision  *float64 `json:"min_trigger_precision,omitempty"`
	MinActionPrecision   *float64 `json:"min_action_precision,omitempty"`
	MinOverturnGain      *float64 `json:"min_overturn_gain,omitempty"`
	MaxP95LatencyDeltaMs *float64 `json:"max_p95_latency_delta_ms,omitempty"`
}

type metaRoutingPolicyEvaluation struct {
	ReplayRecords     int      `json:"replay_records,omitempty"`
	TriggerPrecision  *float64 `json:"trigger_precision,omitempty"`
	ActionPrecision   *float64 `json:"action_precision,omitempty"`
	OverturnGain      *float64 `json:"overturn_gain,omitempty"`
	P95LatencyDeltaMs *float64 `json:"p95_latency_delta_ms,omitempty"`
	Accepted          bool     `json:"accepted,omitempty"`
}

type metaRoutingPolicyBody struct {
	TriggerPolicy  *config.MetaTriggerPolicy     `json:"trigger_policy,omitempty"`
	AllowedActions []config.MetaRefinementAction `json:"allowed_actions,omitempty"`
}
