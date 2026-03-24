package extproc

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func createMetaRoutingPolicyProvider(cfg *config.RouterConfig) (metaRoutingPolicyProvider, error) {
	if cfg == nil || !cfg.MetaRouting.Enabled() {
		return defaultMetaRoutingPolicyProvider(), nil
	}

	path := strings.TrimSpace(os.Getenv(metaRoutingPolicyArtifactPathEnv))
	if path == "" {
		return defaultMetaRoutingPolicyProvider(), nil
	}

	artifact, err := loadMetaRoutingPolicyArtifact(path)
	if err != nil {
		return nil, err
	}
	return artifactMetaRoutingPolicyProvider{artifact: artifact}, nil
}

func loadMetaRoutingPolicyArtifact(path string) (*metaRoutingPolicyArtifact, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read meta-routing policy artifact %q: %w", path, err)
	}

	var artifact metaRoutingPolicyArtifact
	if err := json.Unmarshal(payload, &artifact); err != nil {
		return nil, fmt.Errorf("decode meta-routing policy artifact %q: %w", path, err)
	}

	if err := validateMetaRoutingPolicyArtifact(&artifact); err != nil {
		return nil, fmt.Errorf("validate meta-routing policy artifact %q: %w", path, err)
	}
	return &artifact, nil
}

func validateMetaRoutingPolicyArtifact(artifact *metaRoutingPolicyArtifact) error {
	if artifact == nil {
		return fmt.Errorf("artifact is required")
	}
	if artifact.Version != metaRoutingPolicyArtifactVersion {
		return fmt.Errorf("unsupported artifact version %q", artifact.Version)
	}
	if !isSupportedMetaRoutingArtifactProviderKind(artifact.Provider.Kind) {
		return fmt.Errorf("unsupported provider kind %q", artifact.Provider.Kind)
	}
	if strings.TrimSpace(artifact.Provider.Name) == "" {
		return fmt.Errorf("provider.name is required")
	}
	if strings.TrimSpace(artifact.Provider.Version) == "" {
		return fmt.Errorf("provider.version is required")
	}
	if artifact.FeatureSchema.Name != metaRoutingPolicyFeatureSchemaName {
		return fmt.Errorf("unsupported feature schema name %q", artifact.FeatureSchema.Name)
	}
	if artifact.FeatureSchema.Version != metaRoutingPolicyFeatureSchemaVer {
		return fmt.Errorf("unsupported feature schema version %q", artifact.FeatureSchema.Version)
	}
	if !artifact.Evaluation.Accepted {
		return fmt.Errorf("artifact evaluation must be accepted")
	}
	if artifact.Policy.TriggerPolicy == nil && len(artifact.Policy.AllowedActions) == 0 {
		return fmt.Errorf("artifact policy must define trigger_policy or allowed_actions")
	}
	if err := validateMetaRoutingRolloutGate(artifact.Rollout, artifact.Evaluation); err != nil {
		return err
	}
	return nil
}

func validateMetaRoutingRolloutGate(
	rollout metaRoutingPolicyRolloutGate,
	evaluation metaRoutingPolicyEvaluation,
) error {
	if rollout.MinReplayRecords > 0 && evaluation.ReplayRecords < rollout.MinReplayRecords {
		return fmt.Errorf(
			"replay_records %d below minimum %d",
			evaluation.ReplayRecords,
			rollout.MinReplayRecords,
		)
	}
	if err := validateMetaRoutingMinMetric("trigger_precision", rollout.MinTriggerPrecision, evaluation.TriggerPrecision); err != nil {
		return err
	}
	if err := validateMetaRoutingMinMetric("action_precision", rollout.MinActionPrecision, evaluation.ActionPrecision); err != nil {
		return err
	}
	if err := validateMetaRoutingMinMetric("overturn_gain", rollout.MinOverturnGain, evaluation.OverturnGain); err != nil {
		return err
	}
	if err := validateMetaRoutingMaxMetric("p95_latency_delta_ms", rollout.MaxP95LatencyDeltaMs, evaluation.P95LatencyDeltaMs); err != nil {
		return err
	}
	return nil
}

func validateMetaRoutingMinMetric(name string, min *float64, actual *float64) error {
	if min == nil {
		return nil
	}
	if actual == nil {
		return fmt.Errorf("%s is required when rollout gate is configured", name)
	}
	if *actual < *min {
		return fmt.Errorf("%s %.4f below minimum %.4f", name, *actual, *min)
	}
	return nil
}

func validateMetaRoutingMaxMetric(name string, max *float64, actual *float64) error {
	if max == nil {
		return nil
	}
	if actual == nil {
		return fmt.Errorf("%s is required when rollout gate is configured", name)
	}
	if *actual > *max {
		return fmt.Errorf("%s %.4f above maximum %.4f", name, *actual, *max)
	}
	return nil
}
