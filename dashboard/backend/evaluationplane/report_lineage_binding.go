package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
)

type lineageArtifactRef struct {
	SchemaVersion string `json:"schema_version"`
	Digest        string `json:"digest"`
	MediaType     string `json:"media_type"`
	SizeBytes     int64  `json:"size_bytes"`
}

type lineageWorkload struct {
	SchemaVersion string             `json:"schema_version"`
	ID            string             `json:"id"`
	VisibleCases  lineageArtifactRef `json:"visible_cases"`
	GradingCases  lineageArtifactRef `json:"grading_cases"`
}

type lineagePolicy struct {
	SchemaVersion   string `json:"schema_version"`
	ID              string `json:"id"`
	EntrypointModel string `json:"entrypoint_model"`
	RecipeDigest    string `json:"recipe_digest"`
}

type lineagePool struct {
	SchemaVersion string   `json:"schema_version"`
	ID            string   `json:"id"`
	ArmIDs        []string `json:"arm_ids"`
}

type lineageBinding struct {
	SchemaVersion string `json:"schema_version"`
	ID            string `json:"id"`
	PolicyID      string `json:"policy_id"`
	PoolID        string `json:"pool_id"`
}

type lineageEndpoint struct {
	SchemaVersion  string     `json:"schema_version"`
	URL            string     `json:"url"`
	APIKey         *SecretRef `json:"api_key,omitempty"`
	TimeoutSeconds float64    `json:"timeout_seconds"`
}

type lineageEnvironment struct {
	SchemaVersion              string           `json:"schema_version"`
	ID                         string           `json:"id"`
	TargetID                   string           `json:"target_id"`
	Platform                   string           `json:"platform"`
	HardwareClass              string           `json:"hardware_class"`
	BackendTopologyDigest      string           `json:"backend_topology_digest,omitempty"`
	RouteEval                  *lineageEndpoint `json:"route_eval,omitempty"`
	RoutedChat                 *lineageEndpoint `json:"routed_chat,omitempty"`
	AgentTaskLedger            *lineageEndpoint `json:"agent_task_ledger,omitempty"`
	FaultRecoveryLedger        *lineageEndpoint `json:"fault_recovery_ledger,omitempty"`
	HardPolicyLedger           *lineageEndpoint `json:"hard_policy_ledger,omitempty"`
	ProductionExperimentLedger *lineageEndpoint `json:"production_experiment_ledger,omitempty"`
	Replay                     *lineageEndpoint `json:"replay,omitempty"`
	Currency                   string           `json:"currency"`
}

type lineageExecutor struct {
	SchemaVersion string  `json:"schema_version"`
	TrackID       TrackID `json:"track_id"`
	ExecutorID    string  `json:"executor_id"`
	Mode          Mode    `json:"mode"`
}

func validateLineageBindings(
	runDir string,
	manifest RunManifest,
	document lineageDocument,
	checksums map[string]string,
	executionContract resolvedExecutionContract,
) error {
	resolved := document.Resolved
	recomputedManifestDigest, manifestDigestErr := manifestSemanticDigest(manifest)
	if manifestDigestErr != nil || recomputedManifestDigest != manifest.ManifestDigest {
		return fmt.Errorf("%w: staged manifest digest is invalid", ErrInvalid)
	}
	var claimedManifestDigest string
	if unmarshalErr := json.Unmarshal(resolved["manifest_digest"], &claimedManifestDigest); unmarshalErr != nil || claimedManifestDigest != manifest.ManifestDigest {
		return fmt.Errorf("%w: lineage manifest digest does not match the staged manifest", ErrInvalid)
	}
	var workload lineageWorkload
	if decodeErr := decodeRawStrict(resolved["workload"], &workload); decodeErr != nil {
		return fmt.Errorf("%w: invalid lineage workload: %w", ErrInvalid, decodeErr)
	}
	if workload.SchemaVersion != SchemaVersion {
		return fmt.Errorf("%w: workload snapshot is not bound to the evaluated cases", ErrInvalid)
	}
	if visibleErr := validateRunArtifactRef(runDir, "cases.jsonl", workload.VisibleCases, checksums); visibleErr != nil {
		return fmt.Errorf("%w: visible workload is not bound to evaluated cases: %w", ErrInvalid, visibleErr)
	}
	if gradingErr := validateRunArtifactRef(runDir, "grading-cases.jsonl", workload.GradingCases, checksums); gradingErr != nil {
		return fmt.Errorf("%w: grading workload is not bound to evaluated cases: %w", ErrInvalid, gradingErr)
	}
	workloadIdentity, workloadIdentityErr := canonicalValueDigest(map[string]any{
		"visible_cases": workload.VisibleCases.Digest, "grading_cases": workload.GradingCases.Digest,
	})
	if workloadIdentityErr != nil || workload.ID != "workload-"+strings.TrimPrefix(workloadIdentity, "sha256:")[:16] {
		return fmt.Errorf("%w: workload snapshot id is invalid", ErrInvalid)
	}

	var policy lineagePolicy
	var pool lineagePool
	var binding lineageBinding
	var arms []ModelArm
	var environment lineageEnvironment
	var executors []lineageExecutor
	if decodeErr := decodeRawStrict(resolved["policy"], &policy); decodeErr != nil {
		return decodeErr
	}
	if decodeErr := decodeRawStrict(resolved["pool"], &pool); decodeErr != nil {
		return decodeErr
	}
	if decodeErr := decodeRawStrict(resolved["binding"], &binding); decodeErr != nil {
		return decodeErr
	}
	if decodeErr := decodeRawStrict(resolved["arms"], &arms); decodeErr != nil {
		return decodeErr
	}
	if decodeErr := decodeRawStrict(resolved["environment"], &environment); decodeErr != nil {
		return decodeErr
	}
	if decodeErr := decodeRawStrict(resolved["executors"], &executors); decodeErr != nil {
		return fmt.Errorf("%w: invalid lineage executors: %w", ErrInvalid, decodeErr)
	}
	if validationErr := validateLineageExecutors(manifest, executors); validationErr != nil {
		return validationErr
	}
	identities, lineageErr := validateNormalizedSuiteLineage(
		runDir, manifest, document.NormalizedSuiteIdentities, executionContract.Executor,
	)
	if lineageErr != nil {
		return lineageErr
	}
	if executionContract.Executor.NormalizedSuite {
		if workloadErr := validateNormalizedWorkloadBinding(runDir, manifest, identities); workloadErr != nil {
			return workloadErr
		}
	}
	if validationErr := validateResolvedFactors(
		manifest, policy, pool, binding, arms, environment, identities, executionContract.Executor,
	); validationErr != nil {
		return validationErr
	}
	fixtureRefJSON := bytes.TrimSpace(resolved["fixture_ref"])
	if executionContract.Executor.RequiresFixtureRef {
		var fixtureRef lineageArtifactRef
		if err := decodeRawStrict(resolved["fixture_ref"], &fixtureRef); err != nil || validateCASArtifactRef(runDir, fixtureRef) != nil {
			return fmt.Errorf("%w: replay fixture snapshot is unavailable or unverified", ErrInvalid)
		}
	} else if !bytes.Equal(fixtureRefJSON, []byte("null")) {
		return fmt.Errorf("%w: non-fixture execution cannot claim fixture evidence", ErrInvalid)
	}
	return nil
}

func validateLineageExecutors(manifest RunManifest, executors []lineageExecutor) error {
	expectedExecutor, ok := manifestExecutorIdentity(manifest)
	if !ok || len(executors) != len(manifest.TrackIDs) {
		return fmt.Errorf("%w: lineage executor set does not match the manifest", ErrInvalid)
	}
	for index, executor := range executors {
		if executor.SchemaVersion != SchemaVersion || executor.TrackID != manifest.TrackIDs[index] ||
			executor.ExecutorID != expectedExecutor || executor.Mode != manifest.Mode {
			return fmt.Errorf("%w: lineage executor mapping does not match the manifest", ErrInvalid)
		}
	}
	return nil
}

func validateRunArtifactRef(runDir, name string, ref lineageArtifactRef, checksums map[string]string) error {
	if ref.SchemaVersion != SchemaVersion || !digestPattern.MatchString(ref.Digest) || checksums[name] == "" {
		return fmt.Errorf("artifact ref identity is invalid")
	}
	if err := validateCASArtifactRef(runDir, ref); err != nil {
		return err
	}
	return compareSnapshotToJSONL(runDir, ref, name)
}

func compareSnapshotToJSONL(runDir string, ref lineageArtifactRef, name string) error {
	storeRoot := filepath.Dir(filepath.Dir(runDir))
	hex := strings.TrimPrefix(ref.Digest, "sha256:")
	snapshotBytes, err := readEvidenceBytes(filepath.Join(storeRoot, "objects", "sha256", hex), maxWorkerArtifactBytes)
	if err != nil {
		return err
	}
	var snapshot struct {
		SchemaVersion string            `json:"schema_version"`
		Cases         []json.RawMessage `json:"cases"`
	}
	if snapshotErr := decodeRawStrict(snapshotBytes, &snapshot); snapshotErr != nil || snapshot.SchemaVersion != SchemaVersion {
		return fmt.Errorf("snapshot object is invalid")
	}
	jsonl, err := readEvidenceBytes(filepath.Join(runDir, name), maxWorkerArtifactBytes)
	if err != nil {
		return err
	}
	lines := bytes.Split(bytes.TrimSuffix(jsonl, []byte("\n")), []byte("\n"))
	if len(lines) != len(snapshot.Cases) || (len(lines) == 1 && len(lines[0]) == 0) {
		return fmt.Errorf("snapshot case count does not match JSONL evidence")
	}
	for index := range lines {
		left, leftErr := decodeJSONValue(lines[index])
		right, rightErr := decodeJSONValue(snapshot.Cases[index])
		if leftErr != nil || rightErr != nil || !reflect.DeepEqual(left, right) {
			return fmt.Errorf("snapshot case %d does not match JSONL evidence", index)
		}
	}
	return nil
}

func validateCASArtifactRef(runDir string, ref lineageArtifactRef) error {
	hex := strings.TrimPrefix(ref.Digest, "sha256:")
	if !digestPattern.MatchString(ref.Digest) || ref.SizeBytes < 0 {
		return fmt.Errorf("invalid CAS artifact ref")
	}
	storeRoot := filepath.Dir(filepath.Dir(runDir))
	digest, size, err := privateFileHexDigest(filepath.Join(storeRoot, "objects", "sha256", hex), maxWorkerArtifactBytes)
	if err != nil || digest != hex || size != ref.SizeBytes {
		return fmt.Errorf("CAS artifact ref is unavailable")
	}
	return nil
}

func validateResolvedFactors(
	manifest RunManifest,
	policy lineagePolicy,
	pool lineagePool,
	binding lineageBinding,
	arms []ModelArm,
	environment lineageEnvironment,
	identities *normalizedSuiteIdentityLineage,
	executor executorContract,
) error {
	if policy.SchemaVersion != SchemaVersion || policy.RecipeDigest != manifest.PolicySnapshotDigest ||
		pool.SchemaVersion != SchemaVersion || binding.SchemaVersion != SchemaVersion || environment.SchemaVersion != SchemaVersion ||
		binding.PolicyID != policy.ID || binding.PoolID != pool.ID || environment.TargetID != manifest.Target.ID {
		return fmt.Errorf("%w: resolved factor graph is internally inconsistent", ErrInvalid)
	}
	armIDs := make([]string, len(arms))
	for index := range arms {
		armIDs[index] = arms[index].ID
	}
	if !reflect.DeepEqual(pool.ArmIDs, armIDs) {
		return fmt.Errorf("%w: resolved pool does not bind the declared arms", ErrInvalid)
	}
	switch executor.LineageProfile {
	case lineageRuntime:
		if manifest.Target.Mixture == nil || !sameModelArms(arms, manifest.Target.Mixture.ModelArms) ||
			policy.EntrypointModel != manifest.Target.Mixture.EntrypointModel ||
			pool.ID != stableMixtureFactorID("pool", manifest.Target.Mixture.PoolDigest) ||
			binding.ID != stableMixtureFactorID("binding", manifest.Target.Mixture.BindingDigest) ||
			environment.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
			!sameLineageEndpoint(environment.RouteEval, manifest.Target.RouterAPIURL) ||
			!sameLineageEndpoint(environment.RoutedChat, manifest.Target.EnvoyURL) ||
			!sameFrozenLineageEndpoint(environment.AgentTaskLedger, manifest.Target.AgentTaskLedger) ||
			!sameFrozenLineageEndpoint(environment.FaultRecoveryLedger, manifest.Target.FaultRecoveryLedger) ||
			!sameFrozenLineageEndpoint(environment.HardPolicyLedger, manifest.Target.HardPolicyLedger) ||
			!sameFrozenLineageEndpoint(environment.ProductionExperimentLedger, manifest.Target.ProductionExperimentLedger) || environment.Replay != nil {
			return fmt.Errorf("%w: live factor graph does not match the server-owned target", ErrInvalid)
		}
	case lineageNormalized:
		if identities == nil {
			return fmt.Errorf("%w: replay factor graph omits installed benchmark identities", ErrInvalid)
		}
		if err := validateNormalizedReplayFactors(manifest, policy, pool, binding, arms, environment, *identities); err != nil {
			return fmt.Errorf("%w: replay factor graph does not match the installed benchmark source: %w", ErrInvalid, err)
		}
	case lineageFixture:
		expectedArms, err := builtinFixtureModelArms()
		if err != nil || policy.ID != "fixture-policy" || policy.EntrypointModel != "fixture-entrypoint" ||
			pool.ID != "fixture-pool" || binding.ID != "fixture-binding" || !sameModelArms(arms, expectedArms) ||
			environment.ID != "fixture-environment" || environment.Platform != "local-replay" ||
			environment.HardwareClass != "recorded" || environment.Currency != "USD" ||
			environment.BackendTopologyDigest != "" || environment.RouteEval != nil || environment.RoutedChat != nil ||
			environment.AgentTaskLedger != nil || environment.FaultRecoveryLedger != nil || environment.HardPolicyLedger != nil || environment.ProductionExperimentLedger != nil || environment.Replay != nil {
			return fmt.Errorf("%w: replay factor graph does not match the built-in fixture", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: resolved factor graph lineage contract is invalid", ErrInvalid)
	}
	return nil
}

func stableMixtureFactorID(prefix, digest string) string {
	return prefix + "-" + strings.TrimPrefix(digest, "sha256:")[:16]
}

func builtinFixtureModelArms() ([]ModelArm, error) {
	fastRevision, strongRevision := "fixture-v1", "fixture-v1"
	fastConfig, err := canonicalValueDigest(map[string]any{"fixture": "fast-v1"})
	if err != nil {
		return nil, err
	}
	strongConfig, err := canonicalValueDigest(map[string]any{"fixture": "strong-v1"})
	if err != nil {
		return nil, err
	}
	return []ModelArm{
		{
			ID: "arm-fast", Model: "fixture-fast", ProviderModelIDDigest: digestBytes([]byte("fixture-fast")),
			InputCostPerMillionTokensUSD: 0.5, OutputCostPerMillionTokensUSD: 1,
			Capabilities: []string{"chat"}, Modalities: []string{"text"}, RuntimeRevision: &fastRevision, ConfigDigest: &fastConfig,
		},
		{
			ID: "arm-strong", Model: "fixture-strong", ProviderModelIDDigest: digestBytes([]byte("fixture-strong")),
			InputCostPerMillionTokensUSD: 1.5, OutputCostPerMillionTokensUSD: 3,
			Capabilities: []string{"chat", "vision"}, Modalities: []string{"text", "image"}, RuntimeRevision: &strongRevision, ConfigDigest: &strongConfig,
		},
	}, nil
}

func sameLineageEndpoint(endpoint *lineageEndpoint, expected string) bool {
	if expected == "" {
		return endpoint == nil
	}
	return endpoint != nil && endpoint.SchemaVersion == SchemaVersion && endpoint.URL == expected &&
		endpoint.APIKey == nil && endpoint.TimeoutSeconds == 30
}

func sameFrozenLineageEndpoint(endpoint *lineageEndpoint, expected *ServiceEndpoint) bool {
	if expected == nil {
		return endpoint == nil
	}
	return endpoint != nil && endpoint.SchemaVersion == expected.SchemaVersion && endpoint.URL == expected.URL &&
		reflect.DeepEqual(endpoint.APIKey, expected.APIKey) && endpoint.TimeoutSeconds == expected.TimeoutSeconds
}

func sameModelArms(left, right []ModelArm) bool {
	leftValue, leftErr := modelArmsCanonicalValue(left)
	rightValue, rightErr := modelArmsCanonicalValue(right)
	return leftErr == nil && rightErr == nil && reflect.DeepEqual(leftValue, rightValue)
}

func decodeRawStrict(raw json.RawMessage, destination any) error {
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
