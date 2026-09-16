package evaluationplane

import (
	"fmt"
)

// suiteGateQualification carries server-derived per-track ceilings. Normalized
// imports and replay remain E0; only an exactly qualified server-live declared-
// shift run can contribute a G4 claim.
type suiteGateQualification struct {
	normalizedSuiteRun bool
	commonGateIDs      map[string]struct{}
	suiteTrackIDs      map[string][]TrackID
	suiteTrackLevels   map[string]map[TrackID]EvidenceLevel
}

func (qualification suiteGateQualification) qualifies(gateID string) bool {
	if !qualification.normalizedSuiteRun {
		return false
	}
	_, ok := qualification.commonGateIDs[gateID]
	return ok
}

func (qualification suiteGateQualification) withSealedEvidenceLevels(levels sealedEvidenceLevels) suiteGateQualification {
	qualified := qualification
	qualified.commonGateIDs = make(map[string]struct{}, len(qualification.commonGateIDs))
	for gateID := range qualification.commonGateIDs {
		qualified.commonGateIDs[gateID] = struct{}{}
	}
	if evidenceLevelRank(levels.ByTrack["routing"]) < evidenceLevelRank("E4") {
		delete(qualified.commonGateIDs, "G4")
	}
	return qualified
}

func resolveSuiteGateQualification(
	root string,
	manifest RunManifest,
	executor executorContract,
) (suiteGateQualification, error) {
	if len(manifest.SuiteIDs) == 0 || len(manifest.SuiteExecutors) != len(manifest.SuiteIDs) ||
		len(manifest.SuiteRevisions) != len(manifest.SuiteIDs) {
		return suiteGateQualification{}, fmt.Errorf("%w: installed suite snapshot is incomplete", ErrInvalid)
	}
	seen := make(map[string]struct{}, len(manifest.SuiteIDs))
	for _, suiteID := range manifest.SuiteIDs {
		if _, duplicate := seen[suiteID]; duplicate {
			return suiteGateQualification{}, fmt.Errorf("%w: installed suite snapshot contains a duplicate identity", ErrInvalid)
		}
		seen[suiteID] = struct{}{}
		executorID, present := manifest.SuiteExecutors[suiteID]
		if !present {
			return suiteGateQualification{}, fmt.Errorf("%w: suite executor snapshot is incomplete", ErrInvalid)
		}
		if executorID != executor.ID {
			return suiteGateQualification{}, fmt.Errorf("%w: suite executor snapshot does not match the resolved execution contract", ErrInvalid)
		}
	}
	if !executor.NormalizedSuite {
		return suiteGateQualification{}, nil
	}
	qualification := suiteGateQualification{
		normalizedSuiteRun: executor.NormalizedSuite,
		commonGateIDs:      make(map[string]struct{}),
		suiteTrackIDs:      make(map[string][]TrackID, len(manifest.SuiteIDs)),
		suiteTrackLevels:   make(map[string]map[TrackID]EvidenceLevel, len(manifest.SuiteIDs)),
	}
	liveRoutingSuites := 0
	qualifiedLiveRoutingSuites := 0
	for _, suiteID := range manifest.SuiteIDs {
		document, err := loadInstalledSuiteDocument(root, suiteID)
		if err != nil {
			return suiteGateQualification{}, err
		}
		frozenRevision, present := manifest.SuiteRevisions[suiteID]
		if !present || frozenRevision != document.Manifest.Revision {
			return suiteGateQualification{}, fmt.Errorf("%w: installed suite %q no longer matches the frozen revision", ErrInvalid, suiteID)
		}
		qualification.suiteTrackIDs[suiteID] = append([]TrackID(nil), document.Manifest.TrackIDs...)
		levels := make(map[TrackID]EvidenceLevel, len(document.Manifest.TrackIDs))
		for _, trackID := range document.Manifest.TrackIDs {
			levels[trackID] = "E0"
		}
		qualification.suiteTrackLevels[suiteID] = levels
		if executor.ID == normalizedSuiteLiveExecutorID && containsTrack(document.Manifest.TrackIDs, "routing") {
			liveRoutingSuites++
			eligible, eligibilityErr := installedDeclaredShiftSourceEligible(root, document)
			if eligibilityErr != nil {
				return suiteGateQualification{}, eligibilityErr
			}
			if eligible {
				qualifiedLiveRoutingSuites++
				levels["routing"] = "E4"
			}
		}
	}
	if liveRoutingSuites == 1 && qualifiedLiveRoutingSuites == 1 {
		qualification.commonGateIDs["G4"] = struct{}{}
	}
	return qualification, nil
}
