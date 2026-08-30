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
	SchemaVersion         string           `json:"schema_version"`
	ID                    string           `json:"id"`
	TargetID              string           `json:"target_id"`
	Platform              string           `json:"platform"`
	HardwareClass         string           `json:"hardware_class"`
	BackendTopologyDigest string           `json:"backend_topology_digest,omitempty"`
	RouteEval             *lineageEndpoint `json:"route_eval,omitempty"`
	RoutedChat            *lineageEndpoint `json:"routed_chat,omitempty"`
	Replay                *lineageEndpoint `json:"replay,omitempty"`
	Currency              string           `json:"currency"`
}

func validateLineageBindings(runDir string, manifest RunManifest, resolved map[string]json.RawMessage, checksums map[string]string) error {
	recomputedManifestDigest, err := manifestSemanticDigest(manifest)
	if err != nil || recomputedManifestDigest != manifest.ManifestDigest {
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
	workloadIdentity, err := canonicalValueDigest(map[string]any{
		"visible_cases": workload.VisibleCases.Digest, "grading_cases": workload.GradingCases.Digest,
	})
	if err != nil || workload.ID != "workload-"+strings.TrimPrefix(workloadIdentity, "sha256:")[:16] {
		return fmt.Errorf("%w: workload snapshot id is invalid", ErrInvalid)
	}

	var policy lineagePolicy
	var pool lineagePool
	var binding lineageBinding
	var arms []ModelArm
	var environment lineageEnvironment
	if err := decodeRawStrict(resolved["policy"], &policy); err != nil {
		return err
	}
	if err := decodeRawStrict(resolved["pool"], &pool); err != nil {
		return err
	}
	if err := decodeRawStrict(resolved["binding"], &binding); err != nil {
		return err
	}
	if err := decodeRawStrict(resolved["arms"], &arms); err != nil {
		return err
	}
	if err := decodeRawStrict(resolved["environment"], &environment); err != nil {
		return err
	}
	if err := validateResolvedFactors(manifest, policy, pool, binding, arms, environment); err != nil {
		return err
	}
	if manifest.Mode == ModeReplay {
		var fixtureRef lineageArtifactRef
		if err := decodeRawStrict(resolved["fixture_ref"], &fixtureRef); err != nil || validateCASArtifactRef(runDir, fixtureRef) != nil {
			return fmt.Errorf("%w: replay fixture snapshot is unavailable or unverified", ErrInvalid)
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

func validateResolvedFactors(manifest RunManifest, policy lineagePolicy, pool lineagePool, binding lineageBinding, arms []ModelArm, environment lineageEnvironment) error {
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
	if manifest.Mode == ModeLive {
		if !sameModelArms(arms, manifest.Target.ModelArms) || environment.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
			!sameLineageEndpoint(environment.RouteEval, manifest.Target.RouterAPIURL) ||
			!sameLineageEndpoint(environment.RoutedChat, manifest.Target.EnvoyURL) || environment.Replay != nil {
			return fmt.Errorf("%w: live factor graph does not match the server-owned target", ErrInvalid)
		}
	} else {
		expectedArms, err := builtinFixtureModelArms()
		if err != nil || policy.ID != "fixture-policy" || policy.EntrypointModel != "fixture-entrypoint" ||
			pool.ID != "fixture-pool" || binding.ID != "fixture-binding" || !sameModelArms(arms, expectedArms) ||
			environment.ID != "fixture-environment" || environment.Platform != "local-replay" ||
			environment.HardwareClass != "recorded" || environment.Currency != "USD" ||
			environment.BackendTopologyDigest != "" || environment.RouteEval != nil || environment.RoutedChat != nil || environment.Replay != nil {
			return fmt.Errorf("%w: replay factor graph does not match the built-in fixture", ErrInvalid)
		}
	}
	return nil
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

func sameModelArms(left, right []ModelArm) bool {
	leftValue, leftErr := modelArmsCanonicalValue(left)
	rightValue, rightErr := modelArmsCanonicalValue(right)
	return leftErr == nil && rightErr == nil && reflect.DeepEqual(leftValue, rightValue)
}

func decodeRawStrict(raw json.RawMessage, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
