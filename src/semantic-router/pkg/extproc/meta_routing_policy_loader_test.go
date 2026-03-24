package extproc

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateMetaRoutingPolicyArtifactRejectsUnacceptedArtifact(t *testing.T) {
	artifact := validMetaRoutingPolicyArtifact()
	artifact.Evaluation.Accepted = false

	if err := validateMetaRoutingPolicyArtifact(&artifact); err == nil {
		t.Fatal("expected validation error for unaccepted artifact")
	}
}

func TestCreateMetaRoutingPolicyProviderLoadsArtifact(t *testing.T) {
	path := writeMetaRoutingPolicyArtifact(t, validMetaRoutingPolicyArtifact())
	t.Setenv(metaRoutingPolicyArtifactPathEnv, path)

	provider, err := createMetaRoutingPolicyProvider(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			MetaRouting: config.MetaRoutingConfig{
				Mode: config.MetaRoutingModeShadow,
			},
		},
	})
	if err != nil {
		t.Fatalf("createMetaRoutingPolicyProvider() error = %v", err)
	}

	descriptor := provider.Descriptor()
	if descriptor == nil {
		t.Fatal("expected policy descriptor")
	}
	if descriptor.Kind != metaRoutingPolicyProviderCalibrated {
		t.Fatalf("descriptor.kind = %q, want %q", descriptor.Kind, metaRoutingPolicyProviderCalibrated)
	}
	if descriptor.ArtifactID != "artifact-123" {
		t.Fatalf("descriptor.artifact_id = %q, want artifact-123", descriptor.ArtifactID)
	}
}

func TestArtifactMetaRoutingPolicyProviderOverridesTriggerPolicy(t *testing.T) {
	provider := artifactMetaRoutingPolicyProvider{
		artifact: &metaRoutingPolicyArtifact{
			Version: metaRoutingPolicyArtifactVersion,
			Provider: metaRoutingPolicyArtifactSource{
				Kind:    metaRoutingPolicyProviderCalibrated,
				Name:    "shadow-calibration",
				Version: "2026-03-24",
			},
			FeatureSchema: metaRoutingPolicyFeatureSchema{
				Name:    metaRoutingPolicyFeatureSchemaName,
				Version: metaRoutingPolicyFeatureSchemaVer,
			},
			Evaluation: metaRoutingPolicyEvaluation{
				ReplayRecords:    120,
				TriggerPrecision: metaFloat64Ptr(0.9),
				ActionPrecision:  metaFloat64Ptr(0.82),
				OverturnGain:     metaFloat64Ptr(0.08),
				Accepted:         true,
			},
			Policy: metaRoutingPolicyBody{
				TriggerPolicy: &config.MetaTriggerPolicy{
					DecisionMarginBelow: metaFloat64Ptr(0.2),
				},
			},
		},
	}

	baseCfg := config.MetaRoutingConfig{
		Mode: config.MetaRoutingModeShadow,
		TriggerPolicy: &config.MetaTriggerPolicy{
			DecisionMarginBelow: metaFloat64Ptr(0.05),
		},
		AllowedActions: []config.MetaRefinementAction{{
			Type: config.MetaRoutingActionRerunSignalFamilies,
		}},
	}
	pass := PassTrace{
		DecisionCandidateCount: 2,
		TraceQuality: TraceQuality{
			DecisionMargin: 0.12,
		},
	}

	assessment := provider.Assess(baseCfg, signalEvaluationInput{}, &classification.SignalResults{}, pass)
	if assessment == nil || !assessment.NeedsRefine {
		t.Fatalf("assessment = %+v, want needs_refine=true", assessment)
	}
	assertContainsString(t, assessment.Triggers, metaRoutingTriggerLowDecisionMargin)
}

func validMetaRoutingPolicyArtifact() metaRoutingPolicyArtifact {
	return metaRoutingPolicyArtifact{
		Version:    metaRoutingPolicyArtifactVersion,
		ArtifactID: "artifact-123",
		Provider: metaRoutingPolicyArtifactSource{
			Kind:    metaRoutingPolicyProviderCalibrated,
			Name:    "shadow-calibration",
			Version: "2026-03-24",
		},
		FeatureSchema: metaRoutingPolicyFeatureSchema{
			Name:    metaRoutingPolicyFeatureSchemaName,
			Version: metaRoutingPolicyFeatureSchemaVer,
		},
		Rollout: metaRoutingPolicyRolloutGate{
			MinReplayRecords:     100,
			MinTriggerPrecision:  metaFloat64Ptr(0.8),
			MinActionPrecision:   metaFloat64Ptr(0.75),
			MinOverturnGain:      metaFloat64Ptr(0.05),
			MaxP95LatencyDeltaMs: metaFloat64Ptr(200),
		},
		Evaluation: metaRoutingPolicyEvaluation{
			ReplayRecords:     120,
			TriggerPrecision:  metaFloat64Ptr(0.9),
			ActionPrecision:   metaFloat64Ptr(0.82),
			OverturnGain:      metaFloat64Ptr(0.08),
			P95LatencyDeltaMs: metaFloat64Ptr(90),
			Accepted:          true,
		},
		Policy: metaRoutingPolicyBody{
			TriggerPolicy: &config.MetaTriggerPolicy{
				DecisionMarginBelow: metaFloat64Ptr(0.18),
			},
			AllowedActions: []config.MetaRefinementAction{{
				Type:           config.MetaRoutingActionRerunSignalFamilies,
				SignalFamilies: []string{config.SignalTypeEmbedding},
			}},
		},
	}
}

func writeMetaRoutingPolicyArtifact(t *testing.T, artifact metaRoutingPolicyArtifact) string {
	t.Helper()
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "meta-routing-policy.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}
	return path
}
