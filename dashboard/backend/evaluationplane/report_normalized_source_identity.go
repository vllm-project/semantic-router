package evaluationplane

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

type normalizedSourceRow struct {
	SchemaVersion    string `json:"schema_version"`
	ID               string `json:"id"`
	ArmID            string `json:"arm_id"`
	ActionID         string `json:"action_id"`
	SelectedArmID    string `json:"selected_arm_id"`
	SelectedActionID string `json:"selected_action_id"`
	LeftActionID     string `json:"left_action_id"`
	RightActionID    string `json:"right_action_id"`
	ChosenActionID   string `json:"chosen_action_id"`
}

type normalizedCaseScore struct {
	SourceID string
	Digest   [sha256.Size]byte
}

type normalizedPerturbationSamplingRow struct {
	SchemaVersion   string `json:"schema_version"`
	PairID          string `json:"pair_id"`
	SourceCaseID    string `json:"source_case_id"`
	PerturbedCaseID string `json:"perturbed_case_id"`
}

type installedVisibleCasePlan struct {
	Modality string
	TrackIDs []TrackID
}

func validateNormalizedSourceIdentities(
	suiteRoot string,
	manifest RunManifest,
	identities normalizedSuiteIdentityLineage,
	documents map[string]installedSuiteDocument,
	executor executorContract,
) error {
	expectedCases := make(map[string]struct{})
	expectedArms := make(map[string]struct{})
	expectedActions := make(map[string]struct{})
	for _, suiteID := range manifest.SuiteIDs {
		document, ok := documents[suiteID]
		if !ok {
			return fmt.Errorf("%w: normalized source identity manifest is unavailable", ErrInvalid)
		}
		casePlans, err := installedVisibleCasePlans(suiteRoot, document)
		if err != nil {
			return err
		}
		selected, err := selectedInstalledNormalizedSourceCases(suiteRoot, document, casePlans, manifest)
		if err != nil {
			return err
		}
		for _, sourceID := range selected {
			expectedCases[normalizedSourceKey(suiteID, sourceID)] = struct{}{}
		}
		for _, sourceID := range document.Manifest.ArmIDs {
			if !portableIDPattern.MatchString(sourceID) {
				return fmt.Errorf("%w: normalized suite declares an invalid source arm", ErrInvalid)
			}
			expectedArms[normalizedSourceKey(suiteID, sourceID)] = struct{}{}
		}
		if err := collectInstalledRecordedIdentities(
			suiteRoot, document, expectedArms, expectedActions,
		); err != nil {
			return err
		}
	}
	if !reflect.DeepEqual(expectedCases, normalizedIdentityKeys(identities.CaseIdentities)) {
		return fmt.Errorf("%w: normalized case identities do not match deterministic installed-suite sampling", ErrInvalid)
	}
	if executor.RecordedNormalizedSource &&
		(!reflect.DeepEqual(expectedArms, normalizedIdentityKeys(identities.ArmIdentities)) ||
			!reflect.DeepEqual(expectedActions, normalizedIdentityKeys(identities.ActionIdentities))) {
		return fmt.Errorf("%w: normalized replay identities do not match installed source evidence", ErrInvalid)
	}
	return nil
}

func trackSetsIntersect(left, right []TrackID) bool {
	for _, trackID := range left {
		if containsTrack(right, trackID) {
			return true
		}
	}
	return false
}

func installedVisibleCasePlans(suiteRoot string, document installedSuiteDocument) (map[string]installedVisibleCasePlan, error) {
	plans := make(map[string]installedVisibleCasePlan, document.Manifest.CaseCount)
	err := scanInstalledSuiteRole(suiteRoot, document.Manifest, "visible_cases", true, func(line []byte, lineNumber int) error {
		var row visibleCaseIdentity
		if err := decodeStrictJSONLine(line, &row); err != nil || row.SchemaVersion != SchemaVersion ||
			!portableIDPattern.MatchString(row.ID) || len(row.Messages) == 0 || !validCaseModality(row.Modality) {
			return fmt.Errorf("%w: installed visible case line %d violates the source case contract", ErrInvalid, lineNumber)
		}
		for index, message := range row.Messages {
			if err := validateVisibleMessage(message); err != nil {
				return fmt.Errorf("%w: installed visible case line %d message %d is invalid: %w", ErrInvalid, lineNumber, index+1, err)
			}
		}
		expectedTrackIDs := make([]TrackID, 0, len(document.Manifest.TrackIDs))
		for _, trackID := range document.Manifest.TrackIDs {
			if row.Modality != "text" || trackID != "multimodal" {
				expectedTrackIDs = append(expectedTrackIDs, trackID)
			}
		}
		if len(expectedTrackIDs) == 0 || !reflect.DeepEqual(row.TrackIDs, expectedTrackIDs) {
			return fmt.Errorf("%w: installed visible case line %d track plan does not match its executable suite", ErrInvalid, lineNumber)
		}
		if _, duplicate := plans[row.ID]; duplicate {
			return fmt.Errorf("%w: installed visible cases duplicate a source identity", ErrInvalid)
		}
		plans[row.ID] = installedVisibleCasePlan{
			Modality: row.Modality,
			TrackIDs: append([]TrackID(nil), row.TrackIDs...),
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(plans) != document.Manifest.CaseCount {
		return nil, fmt.Errorf("%w: installed visible case count differs from its manifest", ErrInvalid)
	}
	return plans, nil
}

func selectedNormalizedSourceCases(seed int64, revision string, sourceIDs []string, limit int) []string {
	scores := make([]normalizedCaseScore, len(sourceIDs))
	for index, sourceID := range sourceIDs {
		scores[index] = normalizedCaseScore{
			SourceID: sourceID,
			Digest:   normalizedSourceScore(seed, revision, sourceID),
		}
	}
	sort.Slice(scores, func(left, right int) bool {
		order := bytes.Compare(scores[left].Digest[:], scores[right].Digest[:])
		return order < 0 || (order == 0 && scores[left].SourceID < scores[right].SourceID)
	})
	if limit < len(scores) {
		scores = scores[:limit]
	}
	selected := make([]string, len(scores))
	for index := range scores {
		selected[index] = scores[index].SourceID
	}
	return selected
}

func selectedInstalledNormalizedSourceCases(
	suiteRoot string,
	document installedSuiteDocument,
	casePlans map[string]installedVisibleCasePlan,
	manifest RunManifest,
) ([]string, error) {
	reserved := make(map[string]struct{})
	if containsTrack(manifest.TrackIDs, "routing") {
		pairs, err := installedNormalizedPerturbationPairs(suiteRoot, document)
		if err != nil {
			return nil, err
		}
		if len(pairs) > 0 {
			sort.Slice(pairs, func(left, right int) bool {
				leftDigest := normalizedSourceScore(manifest.Seed, document.Manifest.Revision, pairs[left].PairID)
				rightDigest := normalizedSourceScore(manifest.Seed, document.Manifest.Revision, pairs[right].PairID)
				order := bytes.Compare(leftDigest[:], rightDigest[:])
				return order < 0 || (order == 0 && pairs[left].PairID < pairs[right].PairID)
			})
			pairBudget := manifest.SampleLimit / 2
			if pairBudget > len(pairs) {
				pairBudget = len(pairs)
			}
			for _, pair := range pairs[:pairBudget] {
				reserved[pair.SourceCaseID] = struct{}{}
				reserved[pair.PerturbedCaseID] = struct{}{}
			}
			if manifest.SampleLimit == 1 {
				reserved[pairs[0].SourceCaseID] = struct{}{}
			}
		}
	}
	for sourceID := range reserved {
		if _, exists := casePlans[sourceID]; !exists {
			return nil, fmt.Errorf("%w: normalized perturbation sampling references an unknown installed case", ErrInvalid)
		}
	}

	remaining := manifest.SampleLimit - len(reserved)
	if remaining < 0 {
		remaining = 0
	}
	applicable := make([]string, 0, len(casePlans))
	for sourceID, plan := range casePlans {
		if _, isReserved := reserved[sourceID]; isReserved || !trackSetsIntersect(plan.TrackIDs, manifest.TrackIDs) {
			continue
		}
		applicable = append(applicable, sourceID)
	}
	selected := selectedNormalizedSourceCases(
		manifest.Seed, document.Manifest.Revision, applicable, remaining,
	)
	for sourceID := range reserved {
		selected = append(selected, sourceID)
	}
	sort.Strings(selected)
	return selected, nil
}

func installedNormalizedPerturbationPairs(
	suiteRoot string,
	document installedSuiteDocument,
) ([]normalizedPerturbationSamplingRow, error) {
	var artifacts map[string]json.RawMessage
	if err := json.Unmarshal(document.Manifest.Artifacts, &artifacts); err != nil {
		return nil, fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	if _, exists := artifacts["perturbations"]; !exists {
		return nil, nil
	}
	pairs := []normalizedPerturbationSamplingRow{}
	err := scanInstalledSuiteRole(suiteRoot, document.Manifest, "perturbations", true, func(line []byte, lineNumber int) error {
		var row normalizedPerturbationSamplingRow
		if err := json.Unmarshal(line, &row); err != nil || row.SchemaVersion != normalizedSuiteSchemaVersion ||
			row.PairID == "" || row.SourceCaseID == "" || row.PerturbedCaseID == "" || row.SourceCaseID == row.PerturbedCaseID {
			return fmt.Errorf("%w: installed perturbation line %d has an invalid sampling identity", ErrInvalid, lineNumber)
		}
		pairs = append(pairs, row)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return pairs, nil
}

func normalizedSourceScore(seed int64, revision, sourceID string) [sha256.Size]byte {
	return sha256.Sum256([]byte(fmt.Sprintf("%d\x00%s\x00%s", seed, revision, sourceID)))
}

func collectInstalledRecordedIdentities(
	suiteRoot string,
	document installedSuiteDocument,
	arms map[string]struct{},
	actions map[string]struct{},
) error {
	for _, role := range []string{"outcomes", "decisions", "preferences", "trajectories"} {
		err := scanInstalledSuiteRole(suiteRoot, document.Manifest, role, false, func(line []byte, lineNumber int) error {
			var row normalizedSourceRow
			if err := json.Unmarshal(line, &row); err != nil || row.SchemaVersion != normalizedSuiteSchemaVersion {
				return fmt.Errorf("%w: installed %s line %d has an invalid source identity", ErrInvalid, role, lineNumber)
			}
			armIDs, actionIDs, err := recordedSourceIDs(role, row)
			if err != nil {
				return fmt.Errorf("%w: installed %s line %d: %w", ErrInvalid, role, lineNumber, err)
			}
			for _, sourceID := range armIDs {
				arms[normalizedSourceKey(document.Manifest.ID, sourceID)] = struct{}{}
			}
			for _, sourceID := range actionIDs {
				actions[normalizedSourceKey(document.Manifest.ID, sourceID)] = struct{}{}
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func recordedSourceIDs(role string, row normalizedSourceRow) ([]string, []string, error) {
	var arms, actions []string
	switch role {
	case "outcomes":
		if row.ArmID == "" {
			return nil, nil, fmt.Errorf("outcome omits arm_id")
		}
		arms = append(arms, row.ArmID)
		actions = appendOptionalSourceID(actions, row.ActionID)
	case "decisions":
		arms = appendOptionalSourceID(arms, row.SelectedArmID)
		actions = appendOptionalSourceID(actions, row.SelectedActionID)
	case "preferences":
		if row.LeftActionID == "" || row.RightActionID == "" {
			return nil, nil, fmt.Errorf("preference omits an action identity")
		}
		actions = append(actions, row.LeftActionID, row.RightActionID)
		actions = appendOptionalSourceID(actions, row.ChosenActionID)
	case "trajectories":
		actions = appendOptionalSourceID(actions, row.SelectedActionID)
	default:
		return nil, nil, fmt.Errorf("unsupported source identity role")
	}
	for _, sourceID := range append(append([]string(nil), arms...), actions...) {
		if !portableIDPattern.MatchString(sourceID) {
			return nil, nil, fmt.Errorf("source identity is not portable")
		}
	}
	return arms, actions, nil
}

func appendOptionalSourceID(values []string, sourceID string) []string {
	if sourceID != "" {
		return append(values, sourceID)
	}
	return values
}

func scanInstalledSuiteRole(
	suiteRoot string,
	manifest suiteManifestProjection,
	role string,
	required bool,
	visit func([]byte, int) error,
) error {
	var artifacts map[string]json.RawMessage
	if err := json.Unmarshal(manifest.Artifacts, &artifacts); err != nil {
		return fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	encoded, exists := artifacts[role]
	if !exists {
		if required {
			return fmt.Errorf("%w: normalized suite omits required source identity artifact %q", ErrInvalid, role)
		}
		return nil
	}
	var ref suiteArtifactReference
	if err := decodeExactJSON(encoded, &ref); err != nil {
		return fmt.Errorf("%w: normalized suite source identity artifact is invalid", ErrInvalid)
	}
	domain := "grading"
	switch role {
	case "visible_cases":
		domain = "visible"
	case "media_manifest":
		domain = "metadata"
	}
	path := filepath.Join(suiteRoot, "objects", domain, "sha256", strings.TrimPrefix(ref.Digest, "sha256:"))
	return scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, maxRecordsPerRun, visit)
}

func normalizedSourceKey(suiteID, sourceID string) string {
	return suiteID + "\x00" + sourceID
}

func normalizedIdentityKeys(rows []normalizedLineageIdentity) map[string]struct{} {
	keys := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		keys[normalizedSourceKey(row.SuiteID, row.SourceID)] = struct{}{}
	}
	return keys
}
