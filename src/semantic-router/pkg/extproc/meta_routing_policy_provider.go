package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const metaRoutingPolicyArtifactPathEnv = "VLLM_SR_META_ROUTING_POLICY_PATH"

type MetaRoutingPolicyDescriptor struct {
	Kind       string `json:"kind,omitempty"`
	Name       string `json:"name,omitempty"`
	Version    string `json:"version,omitempty"`
	ArtifactID string `json:"artifact_id,omitempty"`
}

type metaRoutingPolicyProvider interface {
	Descriptor() *MetaRoutingPolicyDescriptor
	Assess(config.MetaRoutingConfig, signalEvaluationInput, *classification.SignalResults, PassTrace) *MetaAssessment
	Plan(config.MetaRoutingConfig, signalEvaluationInput, *classification.SignalResults, *MetaAssessment) *RefinementPlan
}

type deterministicMetaRoutingPolicyProvider struct{}

type artifactMetaRoutingPolicyProvider struct {
	artifact *metaRoutingPolicyArtifact
}

func defaultMetaRoutingPolicyProvider() metaRoutingPolicyProvider {
	return deterministicMetaRoutingPolicyProvider{}
}

func (r *OpenAIRouter) metaRoutingPolicyProvider() metaRoutingPolicyProvider {
	if r != nil && r.MetaRoutingPolicyProvider != nil {
		return r.MetaRoutingPolicyProvider
	}
	return defaultMetaRoutingPolicyProvider()
}

func (deterministicMetaRoutingPolicyProvider) Descriptor() *MetaRoutingPolicyDescriptor {
	return &MetaRoutingPolicyDescriptor{
		Kind:    "deterministic",
		Name:    "built_in",
		Version: "v1",
	}
}

func (deterministicMetaRoutingPolicyProvider) Assess(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	pass PassTrace,
) *MetaAssessment {
	return assessMetaRoutingPass(metaCfg, signalInput, signals, pass)
}

func (deterministicMetaRoutingPolicyProvider) Plan(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	assessment *MetaAssessment,
) *RefinementPlan {
	return buildMetaRoutingPlan(metaCfg, signalInput, signals, assessment)
}

func (p artifactMetaRoutingPolicyProvider) Descriptor() *MetaRoutingPolicyDescriptor {
	if p.artifact == nil {
		return defaultMetaRoutingPolicyProvider().Descriptor()
	}
	return &MetaRoutingPolicyDescriptor{
		Kind:       p.artifact.Provider.Kind,
		Name:       p.artifact.Provider.Name,
		Version:    p.artifact.Provider.Version,
		ArtifactID: p.artifact.ArtifactID,
	}
}

func (p artifactMetaRoutingPolicyProvider) Assess(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	pass PassTrace,
) *MetaAssessment {
	return assessMetaRoutingPass(p.effectiveConfig(metaCfg), signalInput, signals, pass)
}

func (p artifactMetaRoutingPolicyProvider) Plan(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	assessment *MetaAssessment,
) *RefinementPlan {
	return buildMetaRoutingPlan(p.effectiveConfig(metaCfg), signalInput, signals, assessment)
}

func (p artifactMetaRoutingPolicyProvider) effectiveConfig(metaCfg config.MetaRoutingConfig) config.MetaRoutingConfig {
	effective := metaCfg
	if p.artifact == nil {
		return effective
	}
	if p.artifact.Policy.TriggerPolicy != nil {
		effective.TriggerPolicy = cloneMetaTriggerPolicyLocal(p.artifact.Policy.TriggerPolicy)
	}
	if len(p.artifact.Policy.AllowedActions) > 0 {
		effective.AllowedActions = cloneMetaActionsLocal(p.artifact.Policy.AllowedActions)
	}
	return effective
}

func cloneMetaTriggerPolicyLocal(input *config.MetaTriggerPolicy) *config.MetaTriggerPolicy {
	if input == nil {
		return nil
	}
	output := &config.MetaTriggerPolicy{
		DecisionMarginBelow:      cloneFloatPtrLocal(input.DecisionMarginBelow),
		ProjectionBoundaryWithin: cloneFloatPtrLocal(input.ProjectionBoundaryWithin),
		PartitionConflict:        cloneBoolPtrLocal(input.PartitionConflict),
	}
	if len(input.RequiredFamilies) > 0 {
		output.RequiredFamilies = make([]config.MetaRequiredSignalFamily, 0, len(input.RequiredFamilies))
		for _, family := range input.RequiredFamilies {
			output.RequiredFamilies = append(output.RequiredFamilies, config.MetaRequiredSignalFamily{
				Type:          family.Type,
				MinConfidence: cloneFloatPtrLocal(family.MinConfidence),
				MinMatches:    cloneIntPtrLocal(family.MinMatches),
			})
		}
	}
	if len(input.FamilyDisagreements) > 0 {
		output.FamilyDisagreements = append([]config.MetaSignalFamilyDisagreement(nil), input.FamilyDisagreements...)
	}
	return output
}

func cloneMetaActionsLocal(input []config.MetaRefinementAction) []config.MetaRefinementAction {
	if len(input) == 0 {
		return nil
	}
	output := make([]config.MetaRefinementAction, 0, len(input))
	for _, action := range input {
		output = append(output, config.MetaRefinementAction{
			Type:           action.Type,
			SignalFamilies: append([]string(nil), action.SignalFamilies...),
		})
	}
	return output
}

func cloneFloatPtrLocal(value *float64) *float64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneIntPtrLocal(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneBoolPtrLocal(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func isSupportedMetaRoutingArtifactProviderKind(kind string) bool {
	switch strings.TrimSpace(strings.ToLower(kind)) {
	case metaRoutingPolicyProviderCalibrated, metaRoutingPolicyProviderLearned:
		return true
	default:
		return false
	}
}
