package evaluationplane

import (
	"fmt"
	"reflect"
)

type targetFeature string

const (
	targetFeatureTopology                   targetFeature = "topology"
	targetFeatureRouterAPI                  targetFeature = "router-api"
	targetFeatureEnvoyChat                  targetFeature = "envoy-chat"
	targetFeatureMixturePool                targetFeature = "mixture-pool"
	targetFeatureMultimodalArm              targetFeature = "multimodal-arm"
	targetFeatureAgenticLedger              targetFeature = "agentic-ledger"
	targetFeatureAgentTaskLedger            targetFeature = "agent_task_ledger"
	targetFeatureFaultRecoveryLedger        targetFeature = "fault_recovery_ledger"
	targetFeatureHardPolicyLedger           targetFeature = "hard_policy_ledger"
	targetFeatureProductionExperimentLedger targetFeature = "production_experiment_ledger"
)

func recordedTrackRequirements() map[TrackID][]targetFeature {
	requirements := make(map[TrackID][]targetFeature, len(allTrackIDs))
	for _, trackID := range allTrackIDs {
		requirements[trackID] = []targetFeature{}
	}
	return requirements
}

func runtimeTrackRequirements() map[TrackID][]targetFeature {
	return map[TrackID][]targetFeature{
		"routing":    {targetFeatureTopology, targetFeatureRouterAPI, targetFeatureEnvoyChat},
		"model_pool": {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureMixturePool},
		"joint":      {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureMixturePool},
		"agentic":    {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureAgenticLedger},
		"multimodal": {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureMultimodalArm},
		"preference": {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureProductionExperimentLedger},
		"safety":     {targetFeatureTopology, targetFeatureEnvoyChat, targetFeatureHardPolicyLedger},
		"capacity":   {targetFeatureTopology, targetFeatureEnvoyChat},
	}
}

func validateTargetCapabilities(target targetDefinition) error {
	if err := validateMixtureContract(target.Mixture); err != nil {
		return fmt.Errorf("invalid evaluation target registration %q: %w", target.Public.ID, err)
	}
	if target.Mixture != nil && (target.Public.Kind != "mixture-of-models" ||
		target.Public.Mixture == nil || !reflect.DeepEqual(target.Public.Mixture, catalogMixtureFromManifest(target.Mixture))) {
		return fmt.Errorf("invalid evaluation target registration %q: mixture identity diverges from public catalog", target.Public.ID)
	}
	if target.Mixture == nil && target.Public.Mixture != nil {
		return fmt.Errorf("invalid evaluation target registration %q: public mixture has no sealed target", target.Public.ID)
	}
	if len(target.Contract.TrackRequirements) == 0 {
		return fmt.Errorf("invalid evaluation target registration %q: no track capabilities", target.Public.ID)
	}
	for trackID, required := range target.Contract.TrackRequirements {
		if !containsTrack(allTrackIDs, trackID) {
			return fmt.Errorf("invalid evaluation target registration %q: unknown track %q", target.Public.ID, trackID)
		}
		seen := make(map[targetFeature]bool, len(required))
		for _, feature := range required {
			if !portableIDPattern.MatchString(string(feature)) || seen[feature] {
				return fmt.Errorf("invalid evaluation target registration %q: malformed track capability", target.Public.ID)
			}
			seen[feature] = true
		}
	}
	seenFeatures := make(map[targetFeature]bool, len(target.Features))
	for _, feature := range target.Features {
		if !portableIDPattern.MatchString(string(feature)) || seenFeatures[feature] || isEndpointFeature(feature) {
			return fmt.Errorf("invalid evaluation target registration %q: malformed provided feature", target.Public.ID)
		}
		seenFeatures[feature] = true
	}
	if err := validateServiceEndpoint("hard_policy_ledger", target.HardPolicyLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("agent_task_ledger", target.AgentTaskLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("fault_recovery_ledger", target.FaultRecoveryLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("production_experiment_ledger", target.ProductionExperimentLedger); err != nil {
		return err
	}
	if err := validateDistinctTargetCredentials(map[string]*SecretRef{
		"router_api_key":               target.RouterAPIKey,
		"envoy_api_key":                target.EnvoyAPIKey,
		"agent_task_ledger":            endpointSecretRef(target.AgentTaskLedger),
		"fault_recovery_ledger":        endpointSecretRef(target.FaultRecoveryLedger),
		"hard_policy_ledger":           endpointSecretRef(target.HardPolicyLedger),
		"production_experiment_ledger": endpointSecretRef(target.ProductionExperimentLedger),
	}); err != nil {
		return err
	}
	return nil
}

func endpointSecretRef(endpoint *ServiceEndpoint) *SecretRef {
	if endpoint == nil {
		return nil
	}
	return endpoint.APIKey
}

func isEndpointFeature(feature targetFeature) bool {
	return feature == targetFeatureAgentTaskLedger || feature == targetFeatureFaultRecoveryLedger || feature == targetFeatureHardPolicyLedger || feature == targetFeatureProductionExperimentLedger
}

func availableTargetTracks(target targetDefinition) []TrackID {
	features := targetFeatures(target)
	tracks := make([]TrackID, 0, len(target.Contract.TrackRequirements))
	for _, trackID := range allTrackIDs {
		required, declared := target.Contract.TrackRequirements[trackID]
		if declared && targetHasFeatures(features, required) {
			tracks = append(tracks, trackID)
		}
	}
	return tracks
}

func targetFeatures(target targetDefinition) map[targetFeature]bool {
	features := make(map[targetFeature]bool, len(target.Features)+6)
	if target.Contract.ExecutionProfile == targetProfileRuntime && target.Mixture == nil {
		// Runtime evaluation is subject-bound. Connectivity alone must never
		// resurrect the removed generic runtime target or advertise work that
		// cannot be attributed to an immutable recipe and pool.
		return features
	}
	for _, feature := range target.Features {
		features[feature] = true
	}
	if digestPattern.MatchString(target.BackendTopologyDigest) {
		features[targetFeatureTopology] = true
	}
	if target.RouterAPIURL != "" {
		features[targetFeatureRouterAPI] = true
	}
	if target.EnvoyURL != "" {
		features[targetFeatureEnvoyChat] = true
	}
	if target.Mixture != nil && len(target.Mixture.ModelArms) >= 2 && target.Mixture.EntrypointModel != "" {
		features[targetFeatureMixturePool] = true
	}
	if target.Mixture != nil && hasMultimodalArm(target.Mixture.ModelArms) {
		features[targetFeatureMultimodalArm] = true
	}
	if target.AgentTaskLedger != nil {
		features[targetFeatureAgentTaskLedger] = true
	}
	if target.FaultRecoveryLedger != nil {
		features[targetFeatureFaultRecoveryLedger] = true
	}
	if target.AgentTaskLedger != nil || target.FaultRecoveryLedger != nil {
		features[targetFeatureAgenticLedger] = true
	}
	if target.HardPolicyLedger != nil {
		features[targetFeatureHardPolicyLedger] = true
	}
	if target.ProductionExperimentLedger != nil {
		features[targetFeatureProductionExperimentLedger] = true
	}
	return features
}

func targetHasFeatures(available map[targetFeature]bool, required []targetFeature) bool {
	for _, feature := range required {
		if !available[feature] {
			return false
		}
	}
	return true
}

func canonicalModes(modes []Mode) []Mode {
	selected := make(map[Mode]bool, len(modes))
	for _, mode := range modes {
		selected[mode] = true
	}
	result := make([]Mode, 0, len(selected))
	for _, mode := range []Mode{ModeReplay, ModeLive} {
		if selected[mode] {
			result = append(result, mode)
		}
	}
	return result
}

func copyTargetDefinition(target targetDefinition) targetDefinition {
	target.Public = copyCatalogTarget(target.Public)
	target.Contract.TrackRequirements = copyTrackRequirements(target.Contract.TrackRequirements)
	target.RouterAPIKey = copySecretRef(target.RouterAPIKey)
	target.EnvoyAPIKey = copySecretRef(target.EnvoyAPIKey)
	target.AgentTaskLedger = copyServiceEndpoint(target.AgentTaskLedger)
	target.FaultRecoveryLedger = copyServiceEndpoint(target.FaultRecoveryLedger)
	target.HardPolicyLedger = copyServiceEndpoint(target.HardPolicyLedger)
	target.ProductionExperimentLedger = copyServiceEndpoint(target.ProductionExperimentLedger)
	target.Mixture = copyManifestMixture(target.Mixture)
	target.Features = append([]targetFeature(nil), target.Features...)
	return target
}

func copyTrackRequirements(source map[TrackID][]targetFeature) map[TrackID][]targetFeature {
	result := make(map[TrackID][]targetFeature, len(source))
	for trackID, required := range source {
		result[trackID] = append([]targetFeature(nil), required...)
	}
	return result
}

func manifestMatchesTargetDefinition(manifest ManifestTarget, target targetDefinition) bool {
	return manifest.SchemaVersion == SchemaVersion &&
		manifest.ID == target.Public.ID && manifest.Kind == target.Public.Kind &&
		manifest.RouterAPIURL == target.RouterAPIURL && manifest.EnvoyURL == target.EnvoyURL &&
		reflect.DeepEqual(manifest.RouterAPIKey, target.RouterAPIKey) &&
		reflect.DeepEqual(manifest.EnvoyAPIKey, target.EnvoyAPIKey) &&
		reflect.DeepEqual(manifest.AgentTaskLedger, target.AgentTaskLedger) &&
		reflect.DeepEqual(manifest.FaultRecoveryLedger, target.FaultRecoveryLedger) &&
		reflect.DeepEqual(manifest.HardPolicyLedger, target.HardPolicyLedger) &&
		reflect.DeepEqual(manifest.ProductionExperimentLedger, target.ProductionExperimentLedger) &&
		reflect.DeepEqual(manifest.Mixture, target.Mixture) &&
		manifest.BackendTopologyDigest == target.BackendTopologyDigest
}

func sameMixturePool(left, right *ManifestMixture) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.PoolDigest == right.PoolDigest && left.FallbackArmID == right.FallbackArmID &&
		sameModelArms(left.ModelArms, right.ModelArms)
}
