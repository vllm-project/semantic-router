package evaluationplane

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
)

type normalizedLineageIdentity struct {
	SuiteID  string `json:"suite_id"`
	OpaqueID string `json:"opaque_id"`
	SourceID string `json:"source_id"`
}

type normalizedSuiteIdentityLineage struct {
	SchemaVersion    string                      `json:"schema_version"`
	SuiteRevisions   map[string]string           `json:"suite_revisions"`
	CaseIdentities   []normalizedLineageIdentity `json:"case_identities"`
	ArmIdentities    []normalizedLineageIdentity `json:"arm_identities"`
	ActionIdentities []normalizedLineageIdentity `json:"action_identities"`
}

func normalizedOpaqueID(prefix, revision, kind, sourceID string) string {
	digest := sha256.Sum256([]byte(revision + "\x00" + kind + "\x00" + sourceID))
	return fmt.Sprintf("%s-%x", prefix, digest[:12])
}

func validateNormalizedSuiteLineage(
	runDir string,
	manifest RunManifest,
	raw json.RawMessage,
	executor executorContract,
) (*normalizedSuiteIdentityLineage, error) {
	if !executor.NormalizedSuite {
		if len(raw) != 0 {
			return nil, fmt.Errorf("%w: non-normalized run cannot contain normalized suite identities", ErrInvalid)
		}
		return nil, nil
	}
	if len(raw) == 0 {
		return nil, fmt.Errorf("%w: normalized suite run omits private identity lineage", ErrInvalid)
	}
	var identities normalizedSuiteIdentityLineage
	if err := decodeRawStrict(raw, &identities); err != nil || identities.SchemaVersion != normalizedSuiteSchemaVersion ||
		!reflect.DeepEqual(identities.SuiteRevisions, manifest.SuiteRevisions) {
		return nil, fmt.Errorf("%w: normalized suite identity contract is invalid", ErrInvalid)
	}
	suiteRoot := filepath.Join(filepath.Dir(filepath.Dir(runDir)), "suites")
	documents, loadErr := loadInstalledLineageSuites(suiteRoot, manifest)
	if loadErr != nil {
		return nil, loadErr
	}
	if validationErr := validateNormalizedIdentityRows(identities.CaseIdentities, identities.SuiteRevisions, "case", "case"); validationErr != nil {
		return nil, validationErr
	}
	if validationErr := validateNormalizedIdentityRows(identities.ArmIdentities, identities.SuiteRevisions, "arm", "arm"); validationErr != nil {
		return nil, validationErr
	}
	if validationErr := validateNormalizedIdentityRows(identities.ActionIdentities, identities.SuiteRevisions, "action", "action"); validationErr != nil {
		return nil, validationErr
	}
	caseIDs, caseErr := normalizedVisibleCaseIDs(runDir)
	if caseErr != nil {
		return nil, caseErr
	}
	lineageCaseIDs := make([]string, len(identities.CaseIdentities))
	for index := range identities.CaseIdentities {
		lineageCaseIDs[index] = identities.CaseIdentities[index].OpaqueID
	}
	sort.Strings(lineageCaseIDs)
	if !reflect.DeepEqual(lineageCaseIDs, caseIDs) {
		return nil, fmt.Errorf("%w: normalized case identities do not bind the executed workload", ErrInvalid)
	}
	if !executor.RecordedNormalizedSource && (len(identities.ArmIdentities) != 0 || len(identities.ActionIdentities) != 0) {
		return nil, fmt.Errorf("%w: target execution cannot import historical arm or action identities", ErrInvalid)
	}
	if err := validateNormalizedSourceIdentities(suiteRoot, manifest, identities, documents, executor); err != nil {
		return nil, err
	}
	return &identities, nil
}

func loadInstalledLineageSuites(suiteRoot string, manifest RunManifest) (map[string]installedSuiteDocument, error) {
	if len(manifest.SuiteIDs) != len(manifest.SuiteRevisions) {
		return nil, fmt.Errorf("%w: normalized suite manifest identity set is incomplete", ErrInvalid)
	}
	documents := make(map[string]installedSuiteDocument, len(manifest.SuiteIDs))
	for _, suiteID := range manifest.SuiteIDs {
		document, err := loadInstalledSuiteDocument(suiteRoot, suiteID)
		if err != nil || document.Manifest.Revision != manifest.SuiteRevisions[suiteID] {
			return nil, fmt.Errorf("%w: normalized lineage is not bound to an installed private suite manifest", ErrInvalid)
		}
		documents[suiteID] = document
	}
	return documents, nil
}

func validateNormalizedIdentityRows(rows []normalizedLineageIdentity, revisions map[string]string, kind, prefix string) error {
	seenSources := make(map[string]bool, len(rows))
	seenOpaque := make(map[string]bool, len(rows))
	for _, row := range rows {
		revision, exists := revisions[row.SuiteID]
		sourceKey := row.SuiteID + "\x00" + row.SourceID
		if !exists || row.SourceID == "" || seenSources[sourceKey] || seenOpaque[row.OpaqueID] ||
			row.OpaqueID != normalizedOpaqueID(prefix, revision, kind, row.SourceID) {
			return fmt.Errorf("%w: normalized %s identity is invalid or duplicated", ErrInvalid, kind)
		}
		seenSources[sourceKey], seenOpaque[row.OpaqueID] = true, true
	}
	return nil
}

func normalizedVisibleCaseIDs(runDir string) ([]string, error) {
	data, err := readEvidenceBytes(filepath.Join(runDir, "cases.jsonl"), maxWorkerArtifactBytes)
	if err != nil {
		return nil, err
	}
	lines := bytes.Split(bytes.TrimSuffix(data, []byte("\n")), []byte("\n"))
	ids := make([]string, 0, len(lines))
	seen := make(map[string]bool, len(lines))
	for _, line := range lines {
		value, decodeErr := decodeJSONValue(line)
		object, objectOK := value.(map[string]any)
		id, idOK := object["id"].(string)
		schema, schemaOK := object["schema_version"].(string)
		if decodeErr != nil || !objectOK || !idOK || !schemaOK || schema != SchemaVersion ||
			!portableIDPattern.MatchString(id) || seen[id] {
			return nil, fmt.Errorf("%w: normalized visible case identity is invalid", ErrInvalid)
		}
		seen[id] = true
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("%w: normalized workload has no visible cases", ErrInvalid)
	}
	sort.Strings(ids)
	return ids, nil
}

func validateNormalizedReplayFactors(manifest RunManifest, policy lineagePolicy, pool lineagePool, binding lineageBinding, arms []ModelArm, environment lineageEnvironment, identities normalizedSuiteIdentityLineage) error {
	revisionPairs := make([][]string, 0, len(manifest.SuiteRevisions))
	for suiteID, revision := range manifest.SuiteRevisions {
		revisionPairs = append(revisionPairs, []string{suiteID, revision})
	}
	sort.Slice(revisionPairs, func(i, j int) bool { return revisionPairs[i][0] < revisionPairs[j][0] })
	identityDigest, err := canonicalValueDigest(revisionPairs)
	if err != nil {
		return err
	}
	if policy.ID != normalizedOpaqueID("policy", identityDigest, "policy", "normalized-replay") ||
		policy.EntrypointModel != "normalized-replay" ||
		pool.ID != normalizedOpaqueID("pool", identityDigest, "pool", "normalized-replay") ||
		binding.ID != normalizedOpaqueID("binding", identityDigest, "binding", "normalized-replay") ||
		environment.ID != normalizedOpaqueID("environment", identityDigest, "environment", "normalized-replay") ||
		environment.TargetID != manifest.Target.ID || environment.Platform != "normalized-suite-replay" ||
		environment.HardwareClass != "normalized-recorded-evidence" || environment.Currency != "USD" ||
		environment.BackendTopologyDigest != "" || environment.RouteEval != nil || environment.RoutedChat != nil ||
		environment.AgentTaskLedger != nil || environment.FaultRecoveryLedger != nil || environment.HardPolicyLedger != nil || environment.ProductionExperimentLedger != nil || environment.Replay != nil {
		return fmt.Errorf("normalized source factor identities are invalid")
	}

	armIdentities := append([]normalizedLineageIdentity(nil), identities.ArmIdentities...)
	sort.Slice(armIdentities, func(i, j int) bool {
		if armIdentities[i].SuiteID != armIdentities[j].SuiteID {
			return armIdentities[i].SuiteID < armIdentities[j].SuiteID
		}
		return armIdentities[i].SourceID < armIdentities[j].SourceID
	})
	expectedArms := make([]ModelArm, 0, len(armIdentities))
	for _, identity := range armIdentities {
		revision := manifest.SuiteRevisions[identity.SuiteID]
		providerDigest, providerErr := canonicalValueDigest(map[string]any{
			"suite_revision": revision, "source_arm_id": identity.SourceID,
		})
		configDigest, configErr := canonicalValueDigest(map[string]any{
			"kind": "normalized-recorded-arm", "suite_revision": revision, "source_arm_id": identity.SourceID,
		})
		if providerErr != nil || configErr != nil {
			return fmt.Errorf("normalized source arm digest is invalid")
		}
		runtimeRevision, frozenConfig := revision, configDigest
		expectedArms = append(expectedArms, ModelArm{
			ID: identity.OpaqueID, Model: "normalized-replay-" + identity.OpaqueID,
			ProviderModelIDDigest:        providerDigest,
			InputCostPerMillionTokensUSD: 0, OutputCostPerMillionTokensUSD: 0,
			Capabilities: []string{}, Modalities: []string{},
			RuntimeRevision: &runtimeRevision, ConfigDigest: &frozenConfig,
		})
	}
	if !sameModelArms(arms, expectedArms) {
		return fmt.Errorf("normalized source arms do not match private identities")
	}
	return nil
}
