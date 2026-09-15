package evaluationplane

import (
	"fmt"
	"strings"
	"time"
)

// CampaignReadinessRequest pages over the server-owned actor-visible run
// ledger. Optional anchors request exact pair candidates for the current page;
// the browser never uploads or truncates a complete run history.
type CampaignReadinessRequest struct {
	ChangeProfile       ChangeProfile
	Limit               int
	Cursor              string
	BaselineSourceRunID string
	ReferenceRunID      string
}

// CampaignSlotReadiness is an exact server projection of authoring choices.
// The browser consumes identities only and never reimplements evidence ranks,
// executor admission, target reachability, or cohort compatibility.
type CampaignSlotReadiness struct {
	GateID                        string              `json:"gate_id"`
	BindingKind                   CampaignBindingKind `json:"binding_kind"`
	EligibleRunIDs                []string            `json:"eligible_run_ids"`
	ControlledPairSourceRunIDs    []string            `json:"controlled_pair_source_run_ids"`
	ControlledPairCandidateRunIDs []string            `json:"controlled_pair_candidate_run_ids"`
	FidelityReferenceRunIDs       []string            `json:"fidelity_reference_run_ids"`
	FidelityLiveRunIDs            []string            `json:"fidelity_live_run_ids"`
}

type CampaignReadiness struct {
	SchemaVersion string                  `json:"schema_version"`
	ChangeProfile ChangeProfile           `json:"change_profile"`
	NextCursor    string                  `json:"next_cursor,omitempty"`
	TotalRuns     int                     `json:"total_runs"`
	Slots         []CampaignSlotReadiness `json:"slots"`
}

func validateCampaignReadinessRequest(request CampaignReadinessRequest) error {
	if _, ok := campaignProfileContract(request.ChangeProfile); !ok {
		return fmt.Errorf("%w: campaign readiness change_profile is invalid", ErrInvalid)
	}
	if request.Limit < 0 || request.Limit > maxRunPageLimit {
		return fmt.Errorf(
			"%w: campaign readiness limit must be 0 for the default page size or between 1 and %d",
			ErrInvalid, maxRunPageLimit,
		)
	}
	if len(request.Cursor) > maxRunListCursorLength {
		return fmt.Errorf("%w: campaign readiness cursor is too long", ErrInvalid)
	}
	for _, runID := range []string{request.BaselineSourceRunID, request.ReferenceRunID} {
		if runID != "" && !validClientRequestID(runID) {
			return fmt.Errorf("%w: campaign readiness anchor run ID is invalid", ErrInvalid)
		}
	}
	return nil
}

func (s *Service) CampaignReadinessAs(actor Actor, request CampaignReadinessRequest) (CampaignReadiness, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return CampaignReadiness{}, operationErr
	}
	defer releaseOperation()
	if err := validateActor(actor); err != nil {
		return CampaignReadiness{}, err
	}
	if err := validateCampaignReadinessRequest(request); err != nil {
		return CampaignReadiness{}, err
	}
	page, baseline, reference, releaseEvidence, err := s.prepareCampaignReadiness(actor, request)
	if err != nil {
		return CampaignReadiness{}, err
	}
	defer releaseEvidence()
	return s.buildCampaignReadiness(request.ChangeProfile, page, baseline, reference), nil
}

// prepareCampaignReadiness snapshots actor-visible run identities while the
// lifecycle lock is held, then transfers protection to the evidence read lease.
// Deep bundle reads and selected-anchor projection therefore cannot serialize
// unrelated lifecycle reads behind an O(page-size) authoring request.
func (s *Service) prepareCampaignReadiness(
	actor Actor,
	request CampaignReadinessRequest,
) (RunLedger, *Run, *Run, func(), error) {
	s.store.lifecycle.mu.Lock()
	limit := request.Limit
	if limit == 0 {
		limit = defaultRunPageLimit
	}
	page, listErr := s.store.listRunLedger(actor, RunListQuery{Limit: limit, Cursor: request.Cursor})
	if listErr != nil {
		s.store.lifecycle.mu.Unlock()
		return RunLedger{}, nil, nil, nil, listErr
	}
	var baseline, reference *Run
	for _, anchor := range []struct {
		runID string
		set   func(Run)
	}{
		{runID: request.BaselineSourceRunID, set: func(run Run) { baseline = &run }},
		{runID: request.ReferenceRunID, set: func(run Run) { reference = &run }},
	} {
		if anchor.runID == "" {
			continue
		}
		run, err := s.store.runForActorUnlocked(actor, anchor.runID)
		if err != nil {
			s.store.lifecycle.mu.Unlock()
			return RunLedger{}, nil, nil, nil, err
		}
		anchor.set(run)
	}
	releaseEvidence, err := s.acquireEvidenceRead()
	if err != nil {
		s.store.lifecycle.mu.Unlock()
		return RunLedger{}, nil, nil, nil, err
	}
	s.store.lifecycle.mu.Unlock()
	return page, baseline, reference, releaseEvidence, nil
}

func (s *Service) buildCampaignReadiness(
	profileID ChangeProfile,
	page RunLedger,
	baseline *Run,
	reference *Run,
) CampaignReadiness {
	profile, _ := campaignProfileContract(profileID)
	allRuns := append([]Run(nil), page.Runs...)
	for _, anchor := range []*Run{baseline, reference} {
		if anchor != nil && !containsCampaignReadinessRun(allRuns, anchor.ID) {
			allRuns = append(allRuns, *anchor)
		}
	}
	evidence := s.loadCampaignReadinessEvidence(profileID, allRuns)
	sources := s.loadCampaignReadinessSources(profileID, allRuns, evidence)
	result := CampaignReadiness{
		SchemaVersion: SchemaVersion,
		ChangeProfile: profileID,
		NextCursor:    page.NextCursor,
		TotalRuns:     page.TotalRuns,
		Slots:         make([]CampaignSlotReadiness, 0, len(profile.CampaignSlots)),
	}
	for _, slot := range profile.CampaignSlots {
		readiness := CampaignSlotReadiness{
			GateID: slot.GateID, BindingKind: slot.BindingKind,
			EligibleRunIDs:                []string{},
			ControlledPairSourceRunIDs:    []string{},
			ControlledPairCandidateRunIDs: []string{},
			FidelityReferenceRunIDs:       []string{},
			FidelityLiveRunIDs:            []string{},
		}
		if slot.Disposition != GateDispositionNotApplicable {
			switch slot.BindingKind {
			case CampaignBindingRun:
				readiness.EligibleRunIDs = campaignReadyRunIDs(profileID, slot, page.Runs, evidence)
			case CampaignBindingControlledPair:
				readiness.ControlledPairSourceRunIDs = campaignReadyControlledPairSourceRunIDs(page.Runs, sources)
				readiness.ControlledPairCandidateRunIDs = s.campaignReadyControlledPairCandidates(
					profileID, page.Runs, baseline, sources,
				)
			case CampaignBindingFidelityPair:
				readiness.FidelityReferenceRunIDs, readiness.FidelityLiveRunIDs = campaignReadyFidelitySlot(
					profileID, slot, page.Runs, allRuns, reference, evidence,
				)
			}
		}
		result.Slots = append(result.Slots, readiness)
	}
	return result
}

func containsCampaignReadinessRun(runs []Run, runID string) bool {
	for _, run := range runs {
		if run.ID == runID {
			return true
		}
	}
	return false
}

func (s *Service) loadCampaignReadinessEvidence(
	profileID ChangeProfile,
	runs []Run,
) map[string]campaignRunEvidence {
	loaded := make(map[string]campaignRunEvidence, len(runs))
	fidelitySlot, hasFidelity := campaignSlotContract(profileID, "G5")
	hasFidelity = hasFidelity && fidelitySlot.Disposition != GateDispositionNotApplicable
	for _, run := range runs {
		if run.Status != StatusCompleted {
			continue
		}
		item, err := s.loadSealedCampaignRunEvidence(run.ID)
		if err != nil {
			continue
		}
		subjectDigest, err := candidateSubjectDigest(item.manifest, item.report)
		if err != nil {
			continue
		}
		item.anchor.CandidateSubjectDigest = subjectDigest
		if hasFidelity {
			referenceBinding := campaignEvidenceBinding{
				slotID: "g5", gateID: "G5", bindingRole: "reference",
				runID: run.ID, candidate: true,
			}
			liveBinding := referenceBinding
			liveBinding.bindingRole = "live"
			if validateCampaignBoundRun(profileID, referenceBinding, fidelitySlot, item) == nil ||
				validateCampaignBoundRun(profileID, liveBinding, fidelitySlot, item) == nil {
				item, err = s.loadCampaignRunRecords(item)
				if err != nil {
					continue
				}
			}
		}
		loaded[run.ID] = item
	}
	return loaded
}

func (s *Service) loadCampaignReadinessSources(
	profileID ChangeProfile,
	runs []Run,
	evidence map[string]campaignRunEvidence,
) map[string]controlledPairSource {
	loaded := make(map[string]controlledPairSource, len(runs))
	for _, run := range runs {
		if run.Status != StatusCompleted || run.Mode != ModeLive || run.ChangeProfile != profileID {
			continue
		}
		item, ok := evidence[run.ID]
		if !ok || item.attestation == nil {
			continue
		}
		anchorDigest, err := s.store.reportAnchorDigest(run.ID)
		if err != nil {
			continue
		}
		loaded[run.ID] = controlledPairSource{
			run: run, manifest: item.manifest, report: item.report,
			manifestArtifactDigest: item.anchor.ManifestArtifactDigest,
			anchorDigest:           anchorDigest, attestationDigest: item.attestation.Digest,
		}
	}
	return loaded
}

func campaignReadyRunIDs(
	profileID ChangeProfile,
	slot CatalogCampaignSlot,
	runs []Run,
	evidence map[string]campaignRunEvidence,
) []string {
	ready := make([]string, 0, len(runs))
	for _, run := range runs {
		item, ok := evidence[run.ID]
		if !ok {
			continue
		}
		binding := campaignEvidenceBinding{
			slotID: strings.ToLower(slot.GateID), gateID: slot.GateID,
			bindingRole: campaignSingleBindingRole, runID: run.ID, candidate: true,
		}
		if validateCampaignBoundRun(profileID, binding, slot, item) == nil {
			ready = append(ready, run.ID)
		}
	}
	return ready
}

func (s *Service) campaignReadyControlledPairCandidates(
	profileID ChangeProfile,
	pageRuns []Run,
	baseline *Run,
	sources map[string]controlledPairSource,
) []string {
	ready := make([]string, 0)
	if baseline == nil || baseline.ChangeProfile != profileID {
		return ready
	}
	left, ok := sources[baseline.ID]
	if !ok {
		return ready
	}
	registry, err := s.registrySnapshot()
	if err != nil {
		return ready
	}
	for _, candidate := range pageRuns {
		right, ok := sources[candidate.ID]
		if !ok || candidate.ID == baseline.ID || candidate.ChangeProfile != profileID {
			continue
		}
		if validateControlledPairSourcesAgainstRegistry(left, right, s.codeRevision, registry) == nil {
			ready = append(ready, candidate.ID)
		}
	}
	return ready
}

func campaignReadyControlledPairSourceRunIDs(
	runs []Run,
	sources map[string]controlledPairSource,
) []string {
	ready := make([]string, 0, len(runs))
	for _, run := range runs {
		if _, ok := sources[run.ID]; ok {
			ready = append(ready, run.ID)
		}
	}
	return ready
}

type campaignFidelityReadinessFingerprint struct {
	referenceEligible bool
	liveEligible      bool
	cohortDigest      string
	completedAt       *time.Time
	startedAt         time.Time
}

type campaignFidelityReadinessScanner func(campaignRunEvidence, TrackID) (string, error)

type campaignFidelityReadinessCohort struct {
	ChangeProfile         ChangeProfile
	CandidateSubject      string
	SuiteIDs              []string
	TrackIDs              []TrackID
	Seed                  int64
	SampleLimit           int
	WorkloadSnapshot      string
	BenchmarkRevisions    map[string]string
	RecordKeyCohortDigest string
}

func campaignFidelityReadinessRecordDigest(item campaignRunEvidence, trackID TrackID) (string, error) {
	if _, err := campaignAttestedObservations("g5_readiness", item); err != nil {
		return "", err
	}
	records, err := campaignFidelityRecords(item.records, trackID)
	if err != nil || len(records) == 0 {
		return "", fmt.Errorf("fidelity record cohort is unavailable")
	}
	return canonicalValueDigest(campaignFidelityKeys(records))
}

func campaignReadyFidelitySlot(
	profileID ChangeProfile,
	slot CatalogCampaignSlot,
	pageRuns []Run,
	allRuns []Run,
	reference *Run,
	evidence map[string]campaignRunEvidence,
) ([]string, []string) {
	return campaignReadyFidelitySlotWithScanner(
		profileID, slot, pageRuns, allRuns, reference, evidence,
		campaignFidelityReadinessRecordDigest,
	)
}

func campaignReadyFidelitySlotWithScanner(
	profileID ChangeProfile,
	slot CatalogCampaignSlot,
	pageRuns []Run,
	allRuns []Run,
	reference *Run,
	evidence map[string]campaignRunEvidence,
	scan campaignFidelityReadinessScanner,
) ([]string, []string) {
	trackID, trackErr := campaignFidelityTrack(profileID)
	if trackErr != nil || trackID != slot.TrackID {
		return []string{}, []string{}
	}
	fingerprints := make(map[string]campaignFidelityReadinessFingerprint, len(allRuns))
	for _, run := range allRuns {
		item, ok := evidence[run.ID]
		if !ok {
			continue
		}
		referenceBinding := campaignEvidenceBinding{
			slotID: "g5", gateID: "G5", bindingRole: "reference",
			runID: run.ID, candidate: true,
		}
		liveBinding := referenceBinding
		liveBinding.bindingRole = "live"
		fingerprint := campaignFidelityReadinessFingerprint{
			referenceEligible: validateCampaignBoundRun(profileID, referenceBinding, slot, item) == nil,
			liveEligible:      validateCampaignBoundRun(profileID, liveBinding, slot, item) == nil,
		}
		if !fingerprint.referenceEligible && !fingerprint.liveEligible {
			continue
		}
		recordDigest, err := scan(item, trackID)
		if err != nil || item.attestation == nil || item.report.Run.CompletedAt == nil {
			continue
		}
		cohortDigest, err := canonicalValueDigest(campaignFidelityReadinessCohort{
			ChangeProfile: profileID, CandidateSubject: item.anchor.CandidateSubjectDigest,
			SuiteIDs: append([]string(nil), item.report.Run.SuiteIDs...),
			TrackIDs: append([]TrackID(nil), item.report.Run.TrackIDs...),
			Seed:     item.report.Run.Seed, SampleLimit: item.report.Run.SampleLimit,
			WorkloadSnapshot:      item.report.Provenance.WorkloadSnapshotDigest,
			BenchmarkRevisions:    copyCampaignRevisionMap(item.report.Provenance.BenchmarkRevisions),
			RecordKeyCohortDigest: recordDigest,
		})
		if err != nil || item.anchor.CandidateSubjectDigest == "" ||
			item.report.Provenance.WorkloadSnapshotDigest == "" {
			continue
		}
		fingerprint.cohortDigest = cohortDigest
		fingerprint.completedAt = item.report.Run.CompletedAt
		fingerprint.startedAt = item.attestation.StartedAt
		fingerprints[run.ID] = fingerprint
	}
	references := make([]string, 0, len(pageRuns))
	for _, run := range pageRuns {
		fingerprint, ok := fingerprints[run.ID]
		if !ok {
			continue
		}
		if fingerprint.referenceEligible {
			references = append(references, run.ID)
		}
	}
	liveRuns := make([]string, 0, len(pageRuns))
	if reference == nil {
		return references, liveRuns
	}
	left, ok := fingerprints[reference.ID]
	if !ok || !left.referenceEligible || left.completedAt == nil {
		return references, liveRuns
	}
	for _, candidate := range pageRuns {
		right, ok := fingerprints[candidate.ID]
		if !ok || !right.liveEligible || candidate.ID == reference.ID ||
			left.cohortDigest != right.cohortDigest || !right.startedAt.After(left.completedAt.UTC()) {
			continue
		}
		liveRuns = append(liveRuns, candidate.ID)
	}
	return references, liveRuns
}
