package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

func readRunManifestStrict(path string) (RunManifest, []byte, error) {
	data, err := readBundleFile(path)
	if err != nil {
		return RunManifest{}, nil, err
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return RunManifest{}, nil, fmt.Errorf("decode evaluation manifest: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var manifest RunManifest
	if decodeErr := decoder.Decode(&manifest); decodeErr != nil {
		return RunManifest{}, nil, fmt.Errorf("decode evaluation manifest: %w", decodeErr)
	}
	if trailingErr := ensureJSONEOF(decoder); trailingErr != nil {
		return RunManifest{}, nil, trailingErr
	}
	if err := validateRunManifestContract(manifest); err != nil {
		return RunManifest{}, nil, err
	}
	return manifest, data, nil
}

func validateRunManifestContract(manifest RunManifest) error {
	if err := validateRunManifestIdentity(manifest); err != nil {
		return err
	}
	if err := validateRunManifestRevisions(manifest); err != nil {
		return err
	}
	return validateRunManifestTarget(manifest)
}

func validateRunManifestIdentity(manifest RunManifest) error {
	if manifest.SchemaVersion != SchemaVersion || manifest.Target.SchemaVersion != SchemaVersion {
		return fmt.Errorf("evaluation manifest schema_version must be %q", SchemaVersion)
	}
	if !validClientRequestID(manifest.RunID) {
		return fmt.Errorf("evaluation manifest run_id must be a canonical client request UUID")
	}
	if manifest.Name == "" || manifest.Name != strings.TrimSpace(manifest.Name) || len(manifest.Name) > maxRunNameLength ||
		manifest.Description != strings.TrimSpace(manifest.Description) || len(manifest.Description) > maxRunDescriptionLength {
		return fmt.Errorf("evaluation manifest run metadata is invalid")
	}
	if manifest.Mode != ModeReplay && manifest.Mode != ModeLive {
		return fmt.Errorf("evaluation manifest mode is invalid")
	}
	if !safeIDPattern.MatchString(manifest.Target.ID) || !safeIDPattern.MatchString(manifest.Target.Kind) ||
		!validChangeProfile(manifest.ChangeProfile) || !validStoredSuiteIDs(manifest.SuiteIDs) ||
		!validStoredTrackIDs(manifest.TrackIDs) || manifest.SampleLimit < 1 || manifest.SampleLimit > maxSampleLimit ||
		manifest.Concurrency < 1 || manifest.Concurrency > maxRunConcurrency ||
		manifest.Seed < 0 || manifest.Seed > 1<<32-1 || manifest.CreatedAt.IsZero() {
		return fmt.Errorf("evaluation manifest execution identity is invalid")
	}
	if manifest.BaselineRunID != "" && !validClientRequestID(manifest.BaselineRunID) {
		return fmt.Errorf("evaluation manifest baseline_run_id is invalid")
	}
	if err := validateCapacityRunContract(
		manifest.Mode,
		manifest.TrackIDs,
		manifest.Concurrency,
		manifest.CapacitySLO,
		manifest.CapacityLoadProtocol,
	); err != nil {
		return fmt.Errorf("evaluation manifest capacity SLO is invalid: %w", err)
	}
	return nil
}

func validateRunManifestRevisions(manifest RunManifest) error {
	if !digestPattern.MatchString(manifest.ConfigDigest) ||
		!digestPattern.MatchString(manifest.PolicySnapshotDigest) {
		return fmt.Errorf("evaluation manifest configuration identity is invalid")
	}
	if !digestPattern.MatchString(manifest.ManifestDigest) {
		return fmt.Errorf("evaluation manifest manifest_digest is invalid")
	}
	recomputedManifestDigest, err := manifestSemanticDigest(manifest)
	if err != nil || recomputedManifestDigest != manifest.ManifestDigest {
		return fmt.Errorf("evaluation manifest manifest_digest does not match its server-owned semantic value")
	}
	if !sourceRevisionPattern.MatchString(manifest.CodeRevision) {
		return fmt.Errorf("evaluation manifest code_revision is not immutable")
	}
	if manifest.GateContractVersion != GateContractVersion ||
		!validSuiteRevisionSnapshot(manifest.SuiteIDs, manifest.SuiteRevisions) ||
		!validSuiteExecutorSnapshot(manifest.SuiteIDs, manifest.SuiteExecutors) ||
		manifest.RedactionPolicy != "evaluation-default-v1" {
		return fmt.Errorf("evaluation manifest suite and gate contract revisions are invalid")
	}
	return nil
}

func validateRunManifestTarget(manifest RunManifest) error {
	if err := validateAgenticSuiteEndpoints(
		manifest.Mode, manifest.SuiteIDs, manifest.Target.AgentTaskLedger, manifest.Target.FaultRecoveryLedger,
	); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	for name, rawURL := range map[string]string{
		"router_api_url": manifest.Target.RouterAPIURL,
		"envoy_url":      manifest.Target.EnvoyURL,
	} {
		if err := validateServerOrigin(rawURL); err != nil {
			return fmt.Errorf("evaluation manifest target %s is invalid", name)
		}
	}
	if err := validateEndpointCredentialBindings(
		manifest.Target.RouterAPIURL, manifest.Target.EnvoyURL,
		manifest.Target.RouterAPIKey, manifest.Target.EnvoyAPIKey,
	); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateManifestMixtureContract(manifest.Target.Mixture); err != nil {
		return fmt.Errorf("evaluation manifest target mixture is invalid: %w", err)
	}
	arms := []ModelArm(nil)
	if manifest.Target.Mixture != nil {
		arms = manifest.Target.Mixture.ModelArms
		if !targetIDMatchesMixture(manifest.Target.ID, manifest.Target.Mixture.ID) {
			return fmt.Errorf("evaluation manifest target does not identify its frozen Mixture")
		}
		if manifest.PolicySnapshotDigest != manifest.Target.Mixture.RecipeDigest {
			return fmt.Errorf("evaluation manifest policy snapshot does not match its frozen recipe")
		}
	}
	if err := validateTargetContract(
		manifest.Target.RouterAPIKey,
		manifest.Target.EnvoyAPIKey,
		arms,
		manifest.Target.BackendTopologyDigest,
	); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateServiceEndpoint("hard_policy_ledger", manifest.Target.HardPolicyLedger); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateServiceEndpoint("agent_task_ledger", manifest.Target.AgentTaskLedger); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateServiceEndpoint("fault_recovery_ledger", manifest.Target.FaultRecoveryLedger); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateServiceEndpoint("production_experiment_ledger", manifest.Target.ProductionExperimentLedger); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	if err := validateDistinctTargetCredentials(map[string]*SecretRef{
		"router_api_key":               manifest.Target.RouterAPIKey,
		"envoy_api_key":                manifest.Target.EnvoyAPIKey,
		"agent_task_ledger":            endpointSecretRef(manifest.Target.AgentTaskLedger),
		"fault_recovery_ledger":        endpointSecretRef(manifest.Target.FaultRecoveryLedger),
		"hard_policy_ledger":           endpointSecretRef(manifest.Target.HardPolicyLedger),
		"production_experiment_ledger": endpointSecretRef(manifest.Target.ProductionExperimentLedger),
	}); err != nil {
		return fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	return nil
}

func validateManifestMixtureContract(mixture *ManifestMixture) error {
	if mixture != nil && len(mixture.ModelArms) < 2 {
		return fmt.Errorf("mixture requires at least two frozen model arms")
	}
	return validateMixtureContract(mixture)
}

func validSuiteRevisionSnapshot(suiteIDs []string, revisions map[string]string) bool {
	if len(suiteIDs) == 0 || len(revisions) != len(suiteIDs) {
		return false
	}
	seen := make(map[string]bool, len(suiteIDs))
	for _, suiteID := range suiteIDs {
		revision, ok := revisions[suiteID]
		if suiteID == "" || seen[suiteID] || !ok || strings.TrimSpace(revision) == "" || len(revision) > 160 {
			return false
		}
		seen[suiteID] = true
	}
	return true
}

func validSuiteExecutorSnapshot(suiteIDs []string, executors map[string]string) bool {
	if len(suiteIDs) == 0 || len(executors) != len(suiteIDs) {
		return false
	}
	identity := ""
	for _, suiteID := range suiteIDs {
		executor, ok := executors[suiteID]
		if !ok || !portableIDPattern.MatchString(executor) {
			return false
		}
		if identity == "" {
			identity = executor
		} else if identity != executor {
			return false
		}
	}
	return identity != ""
}

func manifestExecutorIdentity(manifest RunManifest) (string, bool) {
	if !validSuiteExecutorSnapshot(manifest.SuiteIDs, manifest.SuiteExecutors) {
		return "", false
	}
	return manifest.SuiteExecutors[manifest.SuiteIDs[0]], true
}
