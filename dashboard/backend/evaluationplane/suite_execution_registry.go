package evaluationplane

import (
	"fmt"
	"strings"
)

func (r *Registry) registerSuite(suite CatalogSuite) error {
	if !portableSuiteIDPattern.MatchString(suite.ID) || suite.Name == "" || suite.Name != strings.TrimSpace(suite.Name) ||
		suite.Description != strings.TrimSpace(suite.Description) || len(suite.TrackIDs) == 0 || !canonicalTrackOrder(suite.TrackIDs) ||
		len(suite.Modes) == 0 || len(suite.Modes) != len(suite.Executors) || evidenceLevelRank(suite.EvidenceLevel) < 0 ||
		suite.Revision == "" || suite.Revision != strings.TrimSpace(suite.Revision) || suite.CaseCount < 0 {
		return fmt.Errorf("invalid evaluation suite registration %q", suite.ID)
	}
	if err := validateCatalogMethods(suite); err != nil {
		return fmt.Errorf("invalid evaluation suite registration %q: %w", suite.ID, err)
	}
	if err := validateCampaignProtocol(suite); err != nil {
		return fmt.Errorf("invalid evaluation suite registration %q: %w", suite.ID, err)
	}
	for _, method := range suite.Methods {
		if method.EvidenceSource == CatalogMethodEvidenceSourceNormalizedImport && suite.EvidenceLevel != "E0" {
			return fmt.Errorf("invalid evaluation suite registration %q: normalized imports must remain E0", suite.ID)
		}
	}
	seenModes := make(map[Mode]struct{}, len(suite.Modes))
	for _, mode := range suite.Modes {
		executorID, present := suite.Executors[mode]
		executor, registered := r.executors[executorID]
		if _, duplicate := seenModes[mode]; duplicate || !present || !registered || executor.Mode != mode {
			return fmt.Errorf("evaluation suite %q declares an incompatible executor for mode %q", suite.ID, mode)
		}
		for _, trackID := range suite.TrackIDs {
			if !containsTrack(executor.TrackIDs, trackID) {
				return fmt.Errorf("evaluation suite %q declares track %q unsupported by executor %q", suite.ID, trackID, executorID)
			}
			track, known := r.tracks[trackID]
			if !known {
				return fmt.Errorf("evaluation suite %q declares unknown track %q", suite.ID, trackID)
			}
			if !containsMode(track.Modes, mode) {
				track.Modes = canonicalModes(append(track.Modes, mode))
				r.tracks[trackID] = track
			}
		}
		seenModes[mode] = struct{}{}
	}
	if _, duplicate := r.suites[suite.ID]; duplicate {
		return fmt.Errorf("duplicate evaluation suite %q", suite.ID)
	}
	r.suites[suite.ID] = copyCatalogSuite(suite)
	r.suiteOrder = append(r.suiteOrder, suite.ID)
	return nil
}

func validateCatalogMethods(suite CatalogSuite) error {
	if len(suite.Methods) == 0 {
		return fmt.Errorf("methods are required")
	}
	seenIDs := make(map[string]struct{}, len(suite.Methods))
	seenTracks := make(map[TrackID]struct{}, len(suite.TrackIDs))
	for _, method := range suite.Methods {
		if !portableSuiteIDPattern.MatchString(method.ID) || !containsTrack(suite.TrackIDs, method.TrackID) || !validCatalogMethodEvidenceSource(method.EvidenceSource) {
			return fmt.Errorf("method identity is invalid")
		}
		if _, duplicate := seenIDs[method.ID]; duplicate {
			return fmt.Errorf("method identities must be unique")
		}
		seenIDs[method.ID] = struct{}{}
		seenTracks[method.TrackID] = struct{}{}
		seenGates := make(map[string]struct{}, len(method.QualifiedGateIDs))
		for _, gateID := range method.QualifiedGateIDs {
			gate, exists := releaseGateDefinitionByID(gateID)
			if !exists || gate.TrackID != method.TrackID || gate.Campaign == nil ||
				gate.Campaign.BindingKind != CampaignBindingRun {
				return fmt.Errorf("method gate is not track-owned")
			}
			if _, duplicate := seenGates[gateID]; duplicate {
				return fmt.Errorf("method gates must be unique")
			}
			seenGates[gateID] = struct{}{}
		}
		switch method.Status {
		case "qualified":
			return fmt.Errorf("qualified methods require server-owned native execution provenance")
		case "configured":
			if method.Reason != "" || (method.EvidenceSource == CatalogMethodEvidenceSourceNormalizedImport && len(method.QualifiedGateIDs) != 0) {
				return fmt.Errorf("configured method readiness is invalid")
			}
			if method.EvidenceSource == CatalogMethodEvidenceSourceServerBrokeredLive &&
				(method.TrackID != "routing" || len(method.QualifiedGateIDs) != 1 || method.QualifiedGateIDs[0] != "G4") {
				return fmt.Errorf("server-brokered declared-shift methods qualify only routing G4")
			}
		case "data_required":
			if strings.TrimSpace(method.Reason) == "" || method.Reason != strings.TrimSpace(method.Reason) {
				return fmt.Errorf("data-required method reason is invalid")
			}
		default:
			return fmt.Errorf("method readiness is invalid")
		}
	}
	if len(seenTracks) != len(suite.TrackIDs) {
		return fmt.Errorf("methods must exactly cover suite tracks")
	}
	return nil
}
