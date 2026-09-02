package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
)

// normalizedWorkloadCase is reconstructed exclusively from an installed suite
// manifest and its private CAS objects. Worker-produced workload snapshots are
// accepted only when every selected row is the exact server reconstruction.
type normalizedWorkloadCase struct {
	Visible visibleCaseIdentity
	Grading gradingCaseEvidence
}

type installedNormalizedWorkload struct {
	Visible map[string]visibleCaseIdentity
	Grading map[string]gradingCaseEvidence
}

func validateNormalizedWorkloadFromLineage(
	runDir string,
	manifest RunManifest,
	executor executorContract,
) error {
	if !executor.NormalizedSuite {
		return nil
	}
	lineageBytes, err := readEvidenceBytes(filepath.Join(runDir, "lineage.json"), maxStructuredArtifactBytes)
	if err != nil {
		return fmt.Errorf("read normalized workload lineage: %w", err)
	}
	document, err := decodeLineageDocument(lineageBytes)
	if err != nil {
		return err
	}
	identities, err := validateNormalizedSuiteLineage(
		runDir, manifest, document.NormalizedSuiteIdentities, executor,
	)
	if err != nil {
		return err
	}
	return validateNormalizedWorkloadBinding(runDir, manifest, identities)
}

func validateNormalizedWorkloadBinding(
	runDir string,
	manifest RunManifest,
	identities *normalizedSuiteIdentityLineage,
) error {
	if identities == nil {
		return fmt.Errorf("%w: normalized workload omits its installed-suite identities", ErrInvalid)
	}
	suiteRoot := filepath.Join(filepath.Dir(filepath.Dir(runDir)), "suites")
	documents, err := loadInstalledLineageSuites(suiteRoot, manifest)
	if err != nil {
		return err
	}
	expected, err := reconstructNormalizedWorkload(suiteRoot, manifest, identities, documents)
	if err != nil {
		return err
	}
	if err := validateNormalizedVisibleRows(filepath.Join(runDir, "cases.jsonl"), expected); err != nil {
		return err
	}
	if err := validateNormalizedGradingRows(filepath.Join(runDir, "grading-cases.jsonl"), expected); err != nil {
		return err
	}
	return nil
}

func reconstructNormalizedWorkload(
	suiteRoot string,
	manifest RunManifest,
	identities *normalizedSuiteIdentityLineage,
	documents map[string]installedSuiteDocument,
) ([]normalizedWorkloadCase, error) {
	orderedIdentities := append([]normalizedLineageIdentity(nil), identities.CaseIdentities...)
	sort.Slice(orderedIdentities, func(left, right int) bool {
		if orderedIdentities[left].SuiteID != orderedIdentities[right].SuiteID {
			return orderedIdentities[left].SuiteID < orderedIdentities[right].SuiteID
		}
		return orderedIdentities[left].SourceID < orderedIdentities[right].SourceID
	})
	if len(orderedIdentities) == 0 {
		return nil, fmt.Errorf("%w: normalized workload sampling selected no cases", ErrInvalid)
	}

	selectedBySuite := make(map[string]map[string]struct{}, len(documents))
	for _, identity := range orderedIdentities {
		selected := selectedBySuite[identity.SuiteID]
		if selected == nil {
			selected = make(map[string]struct{})
			selectedBySuite[identity.SuiteID] = selected
		}
		selected[identity.SourceID] = struct{}{}
	}
	workloads := make(map[string]installedNormalizedWorkload, len(selectedBySuite))
	for suiteID, selected := range selectedBySuite {
		document, exists := documents[suiteID]
		if !exists {
			return nil, fmt.Errorf("%w: normalized workload references an uninstalled suite", ErrInvalid)
		}
		workload, err := loadInstalledNormalizedWorkload(suiteRoot, document, selected)
		if err != nil {
			return nil, err
		}
		workloads[suiteID] = workload
	}

	expected := make([]normalizedWorkloadCase, 0, len(orderedIdentities))
	for _, identity := range orderedIdentities {
		document, documentExists := documents[identity.SuiteID]
		workload, workloadExists := workloads[identity.SuiteID]
		visible, visibleExists := workload.Visible[identity.SourceID]
		grading, gradingExists := workload.Grading[identity.SourceID]
		if !documentExists || !workloadExists || !visibleExists || !gradingExists {
			return nil, fmt.Errorf("%w: normalized workload references a case absent from its installed suite", ErrInvalid)
		}
		expectedCase, err := reconstructNormalizedCase(manifest, document, identity, visible, grading)
		if err != nil {
			return nil, err
		}
		expected = append(expected, expectedCase)
	}
	return expected, nil
}

func loadInstalledNormalizedWorkload(
	suiteRoot string,
	document installedSuiteDocument,
	selected map[string]struct{},
) (installedNormalizedWorkload, error) {
	plans, err := installedVisibleCasePlans(suiteRoot, document)
	if err != nil {
		return installedNormalizedWorkload{}, err
	}
	visible := make(map[string]visibleCaseIdentity, len(selected))
	if err := scanInstalledSuiteRole(suiteRoot, document.Manifest, "visible_cases", true, func(line []byte, lineNumber int) error {
		var row visibleCaseIdentity
		if err := decodeStrictJSONLine(line, &row); err != nil {
			return fmt.Errorf("%w: installed visible case line %d is invalid", ErrInvalid, lineNumber)
		}
		if _, planned := plans[row.ID]; !planned {
			return fmt.Errorf("%w: installed visible workload identity is invalid or duplicated", ErrInvalid)
		}
		if _, wanted := selected[row.ID]; !wanted {
			return nil
		}
		if visible[row.ID].ID != "" {
			return fmt.Errorf("%w: installed visible workload identity is invalid or duplicated", ErrInvalid)
		}
		if row.Tags == nil {
			row.Tags = []string{}
		}
		visible[row.ID] = row
		return nil
	}); err != nil {
		return installedNormalizedWorkload{}, err
	}

	grading := make(map[string]gradingCaseEvidence, len(selected))
	seenGrading := make(map[string]struct{}, len(plans))
	if err := scanInstalledSuiteRole(suiteRoot, document.Manifest, "grading_cases", true, func(line []byte, lineNumber int) error {
		row, err := decodeInstalledGradingCase(line)
		if err != nil {
			return fmt.Errorf("%w: installed grading case line %d is invalid: %w", ErrInvalid, lineNumber, err)
		}
		if _, planned := plans[row.CaseID]; !planned {
			return fmt.Errorf("%w: installed grading workload identity is invalid or duplicated", ErrInvalid)
		}
		if _, duplicate := seenGrading[row.CaseID]; duplicate {
			return fmt.Errorf("%w: installed grading workload identity is invalid or duplicated", ErrInvalid)
		}
		seenGrading[row.CaseID] = struct{}{}
		if _, wanted := selected[row.CaseID]; !wanted {
			return nil
		}
		grading[row.CaseID] = row
		return nil
	}); err != nil {
		return installedNormalizedWorkload{}, err
	}
	if len(seenGrading) != document.Manifest.CaseCount || len(visible) != len(selected) || len(grading) != len(selected) {
		return installedNormalizedWorkload{}, fmt.Errorf("%w: installed visible and grading workloads differ from their manifest", ErrInvalid)
	}
	for sourceID := range visible {
		if grading[sourceID].CaseID == "" {
			return installedNormalizedWorkload{}, fmt.Errorf("%w: installed visible and grading workload identities differ", ErrInvalid)
		}
	}
	return installedNormalizedWorkload{Visible: visible, Grading: grading}, nil
}

func decodeInstalledGradingCase(line []byte) (gradingCaseEvidence, error) {
	var row gradingCaseEvidence
	if err := decodeStrictJSONLine(line, &row); err != nil {
		return gradingCaseEvidence{}, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(line, &fields); err != nil {
		return gradingCaseEvidence{}, err
	}
	if row.SchemaVersion != SchemaVersion || !portableIDPattern.MatchString(row.CaseID) {
		return gradingCaseEvidence{}, fmt.Errorf("grading identity violates its contract")
	}
	if encoded, present := fields["expected_tools"]; present {
		if bytes.Equal(bytes.TrimSpace(encoded), []byte("null")) || row.ExpectedTools == nil {
			return gradingCaseEvidence{}, fmt.Errorf("expected_tools cannot be null")
		}
	} else {
		row.ExpectedTools = []string{}
	}
	if encoded, present := fields["weight"]; present {
		if bytes.Equal(bytes.TrimSpace(encoded), []byte("null")) || !finiteFloat(row.Weight) || row.Weight <= 0 {
			return gradingCaseEvidence{}, fmt.Errorf("weight must be finite and positive")
		}
	} else {
		row.Weight = 1
	}
	return row, nil
}

func reconstructNormalizedCase(
	manifest RunManifest,
	document installedSuiteDocument,
	identity normalizedLineageIdentity,
	visible visibleCaseIdentity,
	grading gradingCaseEvidence,
) (normalizedWorkloadCase, error) {
	projectedTracks := make([]TrackID, 0, len(visible.TrackIDs))
	for _, trackID := range visible.TrackIDs {
		if containsTrack(manifest.TrackIDs, trackID) {
			projectedTracks = append(projectedTracks, trackID)
		}
	}
	if len(projectedTracks) == 0 {
		return normalizedWorkloadCase{}, fmt.Errorf("%w: normalized source case has no selected executable track", ErrInvalid)
	}
	visible.ID = identity.OpaqueID
	visible.TrackIDs = projectedTracks
	if visible.TrajectoryID != nil && *visible.TrajectoryID != "" {
		trajectoryID := normalizedOpaqueID("trajectory", document.Manifest.Revision, "trajectory", *visible.TrajectoryID)
		visible.TrajectoryID = &trajectoryID
	} else {
		visible.TrajectoryID = nil
	}
	grading.CaseID = identity.OpaqueID
	expectedRoute, err := normalizedExpectedArmID(manifest, document.Manifest.Revision, grading.ExpectedRoute)
	if err != nil {
		return normalizedWorkloadCase{}, fmt.Errorf("%w: installed expected_route is not executable: %w", ErrInvalid, err)
	}
	preferredArmID, err := normalizedExpectedArmID(manifest, document.Manifest.Revision, grading.PreferredArmID)
	if err != nil {
		return normalizedWorkloadCase{}, fmt.Errorf("%w: installed preferred_arm_id is not executable: %w", ErrInvalid, err)
	}
	grading.ExpectedRoute = expectedRoute
	grading.PreferredArmID = preferredArmID
	return normalizedWorkloadCase{
		Visible: visible, Grading: grading,
	}, nil
}

func normalizedExpectedArmID(manifest RunManifest, revision string, source *string) (*string, error) {
	if source == nil || *source == "" {
		return nil, nil
	}
	if manifest.Mode == ModeReplay {
		armID := normalizedOpaqueID("arm", revision, "arm", *source)
		return &armID, nil
	}
	if manifest.Mode != ModeLive || manifest.Target.Mixture == nil {
		return nil, fmt.Errorf("normalized arm labels require replay or a frozen live Mixture")
	}
	matched := ""
	for _, arm := range manifest.Target.Mixture.ModelArms {
		if *source != arm.ID && *source != arm.Model {
			continue
		}
		if matched != "" {
			return nil, fmt.Errorf("normalized arm label is ambiguous in the frozen Mixture")
		}
		matched = arm.ID
	}
	if matched == "" {
		return nil, fmt.Errorf("normalized arm label does not identify a frozen Mixture arm")
	}
	return &matched, nil
}

func validateNormalizedVisibleRows(path string, expected []normalizedWorkloadCase) error {
	observed := 0
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, len(expected)+1, func(line []byte, lineNumber int) error {
		if observed >= len(expected) {
			return fmt.Errorf("%w: normalized visible workload contains an unexpected row", ErrInvalid)
		}
		var actual visibleCaseIdentity
		if err := decodeStrictJSONLine(line, &actual); err != nil {
			return fmt.Errorf("%w: normalized visible workload line %d is invalid", ErrInvalid, lineNumber)
		}
		if !sameJSONSemanticValue(expected[observed].Visible, line) {
			return fmt.Errorf("%w: normalized visible workload line %d is not the server-selected installed case", ErrInvalid, lineNumber)
		}
		observed++
		return nil
	})
	if err != nil {
		return err
	}
	if observed != len(expected) {
		return fmt.Errorf("%w: normalized visible workload row count differs from frozen sampling", ErrInvalid)
	}
	return nil
}

func validateNormalizedGradingRows(path string, expected []normalizedWorkloadCase) error {
	observed := 0
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, len(expected)+1, func(line []byte, lineNumber int) error {
		if observed >= len(expected) {
			return fmt.Errorf("%w: normalized grading workload contains an unexpected row", ErrInvalid)
		}
		var actual gradingCaseEvidence
		if err := decodeStrictJSONLine(line, &actual); err != nil {
			return fmt.Errorf("%w: normalized grading workload line %d is invalid", ErrInvalid, lineNumber)
		}
		if !sameJSONSemanticValue(expected[observed].Grading, line) {
			return fmt.Errorf("%w: normalized grading workload line %d is not the server-selected installed label", ErrInvalid, lineNumber)
		}
		observed++
		return nil
	})
	if err != nil {
		return err
	}
	if observed != len(expected) {
		return fmt.Errorf("%w: normalized grading workload row count differs from frozen sampling", ErrInvalid)
	}
	return nil
}

func sameJSONSemanticValue(expected any, actual []byte) bool {
	expectedBytes, err := json.Marshal(expected)
	if err != nil {
		return false
	}
	expectedValue, expectedErr := decodeJSONValue(expectedBytes)
	actualValue, actualErr := decodeJSONValue(actual)
	return expectedErr == nil && actualErr == nil && reflect.DeepEqual(expectedValue, actualValue)
}
