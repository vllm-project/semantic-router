package evaluationplane

import (
	"fmt"
	"math"
	"time"
)

const maxCampaignTimedRequestID = uint64(math.MaxInt64 / int64(time.Second))

type campaignPairedLiveFixture struct {
	baseline  campaignRunEvidence
	candidate campaignRunEvidence
}

type campaignPairedLiveRows struct {
	baselineRecords  []executionRecordEvidence
	candidateRecords []executionRecordEvidence
	baselineEntries  []executionAttestationEntry
	candidateEntries []executionAttestationEntry
}

func newCampaignPairedLiveFixture(caseCount int, candidateFails bool) campaignPairedLiveFixture {
	baselineReport, candidateReport := newCampaignPairedLiveReports(caseCount)
	baselineID, candidateID := baselineReport.Run.ID, candidateReport.Run.ID
	baselineTargetID, candidateTargetID := baselineReport.Run.TargetID, candidateReport.Run.TargetID
	baselinePolicy := baselineReport.Provenance.PolicySnapshotDigest
	candidatePolicy := candidateReport.Provenance.PolicySnapshotDigest

	startedAt := time.Date(2026, time.August, 30, 1, 0, 0, 0, time.UTC)
	baselineManifest := digestBytes([]byte("baseline-live-manifest"))
	candidateManifest := digestBytes([]byte("candidate-live-manifest"))
	baselineManifestArtifact := digestBytes([]byte("baseline-live-manifest-artifact"))
	candidateManifestArtifact := digestBytes([]byte("candidate-live-manifest-artifact"))
	baselineAttestationDigest := digestBytes([]byte("baseline-live-attestation"))
	candidateAttestationDigest := digestBytes([]byte("candidate-live-attestation"))
	rows := newCampaignPairedLiveRows(caseCount, candidateFails)
	baselineRecords := rows.baselineRecords
	candidateRecords := rows.candidateRecords
	baselineEntries := rows.baselineEntries
	candidateEntries := rows.candidateEntries
	baselineAttestation := &executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: baselineID, ManifestDigest: baselineManifest, TargetID: baselineTargetID, Mode: ModeLive,
		PolicySnapshotDigest: baselinePolicy, BackendTopologyDigest: digestBytes([]byte("baseline-topology")),
		StartedAt: startedAt, CompletedAt: startedAt.Add(time.Minute), Entries: baselineEntries,
		Digest: baselineAttestationDigest,
	}
	candidateAttestation := &executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: candidateID, ManifestDigest: candidateManifest, TargetID: candidateTargetID, Mode: ModeLive,
		PolicySnapshotDigest: candidatePolicy, BackendTopologyDigest: digestBytes([]byte("candidate-topology")),
		StartedAt: startedAt.Add(2 * time.Minute), CompletedAt: startedAt.Add(3 * time.Minute), Entries: candidateEntries,
		Digest: candidateAttestationDigest,
	}
	fixture := campaignPairedLiveFixture{
		baseline: campaignRunEvidence{
			report: baselineReport, records: baselineRecords, attestation: baselineAttestation,
			anchor: CampaignEvidenceAnchor{
				SlotID: "g3", GateID: "G3", BindingRole: "baseline", RunID: baselineID,
				ManifestSemanticDigest: baselineManifest, ManifestArtifactDigest: baselineManifestArtifact,
				ReportDigest:               digestBytes([]byte("baseline-live-report")),
				PrivateReceiptDigest:       digestBytes([]byte("baseline-live-private")),
				ExecutionAttestationDigest: baselineAttestationDigest,
			},
		},
		candidate: campaignRunEvidence{
			report: candidateReport, records: candidateRecords, attestation: candidateAttestation,
			anchor: CampaignEvidenceAnchor{
				SlotID: "g3", GateID: "G3", BindingRole: "candidate", RunID: candidateID,
				CandidateSubjectDigest: digestBytes([]byte("candidate-subject")),
				ManifestSemanticDigest: candidateManifest, ManifestArtifactDigest: candidateManifestArtifact,
				ReportDigest:               digestBytes([]byte("candidate-live-report")),
				PrivateReceiptDigest:       digestBytes([]byte("candidate-live-private")),
				ExecutionAttestationDigest: candidateAttestationDigest,
			},
		},
	}
	return sealCampaignControlledPairFixture(fixture)
}

func newCampaignPairedLiveReports(caseCount int) (Report, Report) {
	baselineID := "2650bcb4-05af-46e4-96d4-cb9ec36b393d"
	candidateID := "ce0844b2-d9bb-4a30-b4c9-b9161bc26011"
	baselineTargetID := "baseline--mom-core"
	candidateTargetID := "candidate--mom-core"
	baselinePolicy := digestBytes([]byte("baseline-policy"))
	candidatePolicy := digestBytes([]byte("candidate-policy"))
	poolDigest := digestBytes([]byte("paired-live-pool"))
	bindingDigest := digestBytes([]byte("baseline-binding"))
	selectorDigest := digestString("paired-live-selector")
	mixture := &CatalogMixture{
		ID: "mom-core", RecipeName: "core", RecipeDigest: baselinePolicy,
		PoolDigest: poolDigest, BindingDigest: bindingDigest,
		SelectorPolicyDigest: selectorDigest, SelectorDigest: selectorDigest,
		AdaptationDigest: digestString("paired-live-adaptation"),
	}
	baseline := Report{
		SchemaVersion: SchemaVersion, AttestationRevision: ServerAttestationRevision,
		Run: Run{
			SchemaVersion: SchemaVersion, ID: baselineID, Status: StatusCompleted, Mode: ModeLive,
			EvidenceLevel: "E3", TargetID: baselineTargetID, Mixture: mixture,
			ChangeProfile: "recipe", SuiteIDs: []string{"installed-routing"},
			TrackIDs: []TrackID{"routing"}, SampleLimit: caseCount, Concurrency: 1, Seed: 17,
		},
		Provenance: Provenance{
			SchemaVersion: SchemaVersion, CodeRevision: "code-revision",
			BenchmarkRevisions: map[string]string{
				"installed-routing": digestBytes([]byte("installed-suite-revision")),
			},
			PolicySnapshotDigest: baselinePolicy, BindingSnapshotDigest: bindingDigest,
			PoolSnapshotDigest: poolDigest, WorkloadSnapshotDigest: digestBytes([]byte("paired-live-workload")),
			EnvironmentSnapshotDigest: digestBytes([]byte("paired-live-environment")), TargetID: baselineTargetID, Seed: 17,
		},
	}
	candidate := baseline
	candidate.Run.ID = candidateID
	candidate.Run.BaselineRunID = baselineID
	candidate.Run.TargetID = candidateTargetID
	candidate.Provenance = baseline.Provenance
	candidate.Provenance.BenchmarkRevisions = copyCampaignRevisionMap(baseline.Provenance.BenchmarkRevisions)
	candidate.Provenance.PolicySnapshotDigest = candidatePolicy
	candidate.Provenance.TargetID = candidateTargetID
	candidateMixture := *mixture
	candidateMixture.RecipeDigest = candidatePolicy
	candidate.Run.Mixture = &candidateMixture
	return baseline, candidate
}

func newCampaignPairedLiveRows(caseCount int, candidateFails bool) campaignPairedLiveRows {
	result := campaignPairedLiveRows{
		baselineRecords:  make([]executionRecordEvidence, 0, caseCount),
		candidateRecords: make([]executionRecordEvidence, 0, caseCount),
		baselineEntries:  make([]executionAttestationEntry, 0, caseCount),
		candidateEntries: make([]executionAttestationEntry, 0, caseCount),
	}
	var requestID uint64 = 1
	for index := 0; index < caseCount; index++ {
		caseID := fmt.Sprintf("case-%d", index+1)
		attemptID := "attempt-" + caseID
		recordID := "record-" + caseID
		baselineReceipt := digestBytes([]byte("baseline-" + caseID))
		candidateReceipt := digestBytes([]byte("candidate-" + caseID))
		baselineSuccess := true
		candidateSuccess := !candidateFails
		baselineQuality, candidateQuality := 0.0, 1.0
		baselineArm, candidateArm := "arm-wrong", "arm-right"
		baselineLatency, candidateLatency := 100.0, 95.0
		if candidateFails {
			baselineQuality, baselineArm = 1, "arm-right"
			candidateLatency = 200
		}
		result.baselineRecords = append(result.baselineRecords, executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: recordID, TrackID: "routing", CaseID: caseID,
			AttemptID: attemptID, Status: "succeeded", SelectedArmID: &baselineArm,
			Success: &baselineSuccess, Quality: &baselineQuality, LatencyMS: &baselineLatency,
			BrokerReceipt: &baselineReceipt,
		})
		candidateRecord := executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: recordID, TrackID: "routing", CaseID: caseID,
			AttemptID: attemptID, SelectedArmID: &candidateArm, Success: &candidateSuccess,
			LatencyMS: &candidateLatency, BrokerReceipt: &candidateReceipt,
		}
		candidateEntry := executionAttestationEntry{
			RequestID: requestID, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
			CaseID: caseID, AttemptID: attemptID, BrokerReceipt: candidateReceipt,
			Success: candidateSuccess, LatencyMicroseconds: int64(candidateLatency * 1000),
		}
		if candidateFails {
			candidateRecord.Status = "failed"
			candidateRecord.SelectedArmID = nil
			candidateEntry.SelectedModel = nil
		} else {
			candidateRecord.Status = "succeeded"
			candidateRecord.Quality = &candidateQuality
			candidateEntry.Quality = &candidateQuality
			candidateEntry.SelectedModel = &candidateArm
			candidateEntry.ArmID = &candidateArm
		}
		result.candidateRecords = append(result.candidateRecords, candidateRecord)
		result.baselineEntries = append(result.baselineEntries, executionAttestationEntry{
			RequestID: requestID, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
			CaseID: caseID, AttemptID: attemptID, BrokerReceipt: baselineReceipt,
			Success: true, Quality: &baselineQuality, LatencyMicroseconds: int64(baselineLatency * 1000),
			SelectedModel: &baselineArm, ArmID: &baselineArm,
		})
		result.candidateEntries = append(result.candidateEntries, candidateEntry)
		requestID++
	}
	return result
}

func withCampaignMoMCoreTracks(fixture campaignPairedLiveFixture) campaignPairedLiveFixture {
	poolDigest := fixture.baseline.report.Provenance.PoolSnapshotDigest
	selectorPolicy := digestString("campaign-selector-policy")
	baselineMixture := &CatalogMixture{
		ID: "mom-core", EntrypointModel: "vllm-sr/auto", RecipeName: "core",
		RecipeDigest: fixture.baseline.report.Provenance.PolicySnapshotDigest,
		PoolDigest:   poolDigest, BindingDigest: fixture.baseline.report.Provenance.BindingSnapshotDigest,
		SelectorPolicyDigest: selectorPolicy,
		SelectorDigest:       selectorSnapshotDigest(selectorPolicy, []SupportModel{}),
		AdaptationDigest:     digestString("campaign-adaptation"),
		ModelArms:            []ModelArm{{ID: "arm-right", Model: "model-right"}, {ID: "arm-wrong", Model: "model-wrong"}},
	}
	candidateMixture := *baselineMixture
	candidateMixture.RecipeDigest = fixture.candidate.report.Provenance.PolicySnapshotDigest
	candidateMixture.BindingDigest = fixture.candidate.report.Provenance.BindingSnapshotDigest
	candidateMixture.ModelArms = append([]ModelArm(nil), baselineMixture.ModelArms...)
	fixture.baseline.report.Run.Mixture = baselineMixture
	fixture.candidate.report.Run.Mixture = &candidateMixture
	fixture.baseline.report.Run.TrackIDs = []TrackID{"routing", "model_pool", "joint"}
	fixture.candidate.report.Run.TrackIDs = []TrackID{"routing", "model_pool", "joint"}

	for index := range fixture.baseline.records {
		caseID := fixture.baseline.records[index].CaseID
		for _, arm := range baselineMixture.ModelArms {
			quality := 0.4
			latency := 45.0
			if arm.ID == "arm-right" {
				quality, latency = 0.9, 60
			}
			for _, side := range []struct {
				role     string
				evidence *campaignRunEvidence
			}{
				{role: "baseline", evidence: &fixture.baseline},
				{role: "candidate", evidence: &fixture.candidate},
			} {
				armID := arm.ID
				success := true
				attemptID := "attempt-pool-" + caseID + "-" + arm.ID
				receipt := digestBytes([]byte(side.role + "-pool-" + caseID + "-" + arm.ID))
				side.evidence.records = append(side.evidence.records, executionRecordEvidence{
					SchemaVersion: SchemaVersion, ID: "record-pool-" + caseID + "-" + arm.ID,
					TrackID: "model_pool", CaseID: caseID, AttemptID: attemptID,
					Status: "succeeded", ArmID: &armID, Success: &success, Quality: float64Reference(quality),
					LatencyMS: float64Reference(latency), BrokerReceipt: &receipt,
				})
				side.evidence.attestation.Entries = append(side.evidence.attestation.Entries, executionAttestationEntry{
					RequestID: nextCampaignRequestID(side.evidence.attestation.Entries),
					Operation: workerBrokerArmChatCompletion, TrackID: "model_pool",
					CaseID: caseID, AttemptID: attemptID, ArmID: &armID,
					BrokerReceipt: receipt, Success: true, Quality: float64Reference(quality),
					LatencyMicroseconds: int64(latency * 1000),
				})
			}
		}

		baselineArm, candidateArm := "arm-wrong", "arm-right"
		for _, side := range []struct {
			role     string
			evidence *campaignRunEvidence
			armID    string
			quality  float64
		}{
			{role: "baseline", evidence: &fixture.baseline, armID: baselineArm, quality: 0.4},
			{role: "candidate", evidence: &fixture.candidate, armID: candidateArm, quality: 1.0},
		} {
			armID := side.armID
			success := true
			attemptID := "attempt-joint-" + caseID
			receipt := digestBytes([]byte(side.role + "-joint-" + caseID))
			latency := 80.0
			side.evidence.records = append(side.evidence.records, executionRecordEvidence{
				SchemaVersion: SchemaVersion, ID: "record-joint-" + caseID,
				TrackID: "joint", CaseID: caseID, AttemptID: attemptID,
				Status: "succeeded", SelectedArmID: &armID, Success: &success,
				Quality: float64Reference(side.quality), LatencyMS: &latency, BrokerReceipt: &receipt,
			})
			side.evidence.attestation.Entries = append(side.evidence.attestation.Entries, executionAttestationEntry{
				RequestID: nextCampaignRequestID(side.evidence.attestation.Entries),
				Operation: workerBrokerRoutedChatCompletion, TrackID: "joint",
				CaseID: caseID, AttemptID: attemptID, ArmID: &armID,
				BrokerReceipt: receipt, Success: true, Quality: float64Reference(side.quality),
				LatencyMicroseconds: int64(latency * 1000),
			})
		}
	}
	return sealCampaignControlledPairFixture(fixture)
}

func withChangedCandidatePool(fixture campaignPairedLiveFixture) campaignPairedLiveFixture {
	fixture.baseline.report.Run.ChangeProfile = "model_pool"
	fixture.candidate.report.Run.ChangeProfile = "model_pool"
	fixture.candidate.report.Provenance.PolicySnapshotDigest = fixture.baseline.report.Provenance.PolicySnapshotDigest
	fixture.candidate.attestation.PolicySnapshotDigest = fixture.baseline.attestation.PolicySnapshotDigest
	fixture.candidate.report.Run.Mixture.RecipeDigest = fixture.baseline.report.Run.Mixture.RecipeDigest
	newPoolDigest := digestBytes([]byte("changed-candidate-pool"))
	fixture.candidate.report.Provenance.PoolSnapshotDigest = newPoolDigest
	fixture.candidate.report.Run.Mixture.PoolDigest = newPoolDigest
	fixture.candidate.report.Run.Mixture.ModelArms = append(
		append([]ModelArm(nil), fixture.candidate.report.Run.Mixture.ModelArms...),
		ModelArm{ID: "arm-extra", Model: "model-extra"},
	)
	for index := 0; index < fixture.candidate.report.Run.SampleLimit; index++ {
		caseID := fmt.Sprintf("case-%d", index+1)
		armID := "arm-extra"
		success := true
		quality, latency := 0.9, 55.0
		attemptID := "attempt-pool-" + caseID + "-" + armID
		receipt := digestBytes([]byte("candidate-pool-" + caseID + "-" + armID))
		fixture.candidate.records = append(fixture.candidate.records, executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: "record-pool-" + caseID + "-" + armID,
			TrackID: "model_pool", CaseID: caseID, AttemptID: attemptID,
			Status: "succeeded", ArmID: &armID, Success: &success, Quality: &quality,
			LatencyMS: &latency, BrokerReceipt: &receipt,
		})
		fixture.candidate.attestation.Entries = append(fixture.candidate.attestation.Entries, executionAttestationEntry{
			RequestID: nextCampaignRequestID(fixture.candidate.attestation.Entries),
			Operation: workerBrokerArmChatCompletion, TrackID: "model_pool",
			CaseID: caseID, AttemptID: attemptID, ArmID: &armID,
			BrokerReceipt: receipt, Success: true, Quality: &quality,
			LatencyMicroseconds: int64(latency * 1000),
		})
	}
	return sealCampaignControlledPairFixture(fixture)
}

func withCampaignPoolArm(
	fixture campaignPairedLiveFixture,
	arm ModelArm,
	quality float64,
) campaignPairedLiveFixture {
	fixture.baseline.report.Run.Mixture.ModelArms = append(
		fixture.baseline.report.Run.Mixture.ModelArms,
		arm,
	)
	fixture.candidate.report.Run.Mixture.ModelArms = append(
		fixture.candidate.report.Run.Mixture.ModelArms,
		arm,
	)
	for index := range fixture.baseline.report.Run.SampleLimit {
		caseID := fmt.Sprintf("case-%d", index+1)
		for _, side := range []struct {
			role     string
			evidence *campaignRunEvidence
		}{
			{role: "baseline", evidence: &fixture.baseline},
			{role: "candidate", evidence: &fixture.candidate},
		} {
			armID := arm.ID
			success := true
			latency := 50.0
			attemptID := "attempt-pool-" + caseID + "-" + arm.ID
			receipt := digestBytes([]byte(side.role + "-pool-" + caseID + "-" + arm.ID))
			side.evidence.records = append(side.evidence.records, executionRecordEvidence{
				SchemaVersion: SchemaVersion, ID: "record-pool-" + caseID + "-" + arm.ID,
				TrackID: "model_pool", CaseID: caseID, AttemptID: attemptID,
				Status: "succeeded", ArmID: &armID, Success: &success, Quality: &quality,
				LatencyMS: &latency, BrokerReceipt: &receipt,
			})
			side.evidence.attestation.Entries = append(
				side.evidence.attestation.Entries,
				executionAttestationEntry{
					RequestID: nextCampaignRequestID(side.evidence.attestation.Entries),
					Operation: workerBrokerArmChatCompletion, TrackID: "model_pool",
					CaseID: caseID, AttemptID: attemptID, ArmID: &armID,
					BrokerReceipt: receipt, Success: true, Quality: &quality,
					LatencyMicroseconds: int64(latency * 1000),
				},
			)
		}
	}
	return sealCampaignControlledPairFixture(fixture)
}

func sealCampaignControlledPairFixture(fixture campaignPairedLiveFixture) campaignPairedLiveFixture {
	const sessionID = "23d94ba8-4da7-4936-ac99-a85afc492fad"
	startedAt := time.Date(2026, time.August, 30, 1, 0, 0, 0, time.UTC)
	topologyDigest := digestString("controlled-pair-test-topology")
	fixture.baseline.attestation.BackendTopologyDigest = topologyDigest
	fixture.candidate.attestation.BackendTopologyDigest = topologyDigest
	fixture.baseline.attestation.StartedAt = startedAt
	fixture.candidate.attestation.StartedAt = startedAt
	fixture.baseline.attestation.CompletedAt = startedAt.Add(time.Hour)
	fixture.candidate.attestation.CompletedAt = startedAt.Add(time.Hour)
	fixture.baseline.manifest = RunManifest{
		SchemaVersion: SchemaVersion, RunID: fixture.baseline.report.Run.ID,
		ManifestDigest: fixture.baseline.anchor.ManifestSemanticDigest, Mode: ModeLive,
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: fixture.baseline.report.Run.TargetID,
			RouterAPIURL: "http://baseline-router.test", EnvoyURL: "http://baseline-envoy.test",
			BackendTopologyDigest: topologyDigest,
		},
	}
	fixture.candidate.manifest = RunManifest{
		SchemaVersion: SchemaVersion, RunID: fixture.candidate.report.Run.ID,
		ManifestDigest: fixture.candidate.anchor.ManifestSemanticDigest, Mode: ModeLive,
		BaselineRunID: fixture.baseline.report.Run.ID,
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: fixture.candidate.report.Run.TargetID,
			RouterAPIURL: "http://candidate-router.test", EnvoyURL: "http://candidate-envoy.test",
			BackendTopologyDigest: topologyDigest,
		},
	}
	type entryCoordinate struct {
		trackID   TrackID
		caseID    string
		attemptID string
		operation string
		armID     string
	}
	keyFor := func(entry executionAttestationEntry) entryCoordinate {
		armID := ""
		if entry.Operation == workerBrokerArmChatCompletion {
			armID = stringValue(entry.ArmID)
		}
		return entryCoordinate{
			trackID: entry.TrackID, caseID: entry.CaseID,
			attemptID: entry.AttemptID, operation: entry.Operation, armID: armID,
		}
	}
	baselineEntries := make(map[entryCoordinate]bool, len(fixture.baseline.attestation.Entries))
	candidateEntries := make(map[entryCoordinate]bool, len(fixture.candidate.attestation.Entries))
	for _, entry := range fixture.baseline.attestation.Entries {
		baselineEntries[keyFor(entry)] = true
	}
	for _, entry := range fixture.candidate.attestation.Entries {
		candidateEntries[keyFor(entry)] = true
	}
	seal := func(role string, manifest RunManifest, entries []executionAttestationEntry) {
		for index := range entries {
			entry := &entries[index]
			key := keyFor(*entry)
			coordinate := controlledPairCoordinate(key)
			paired := baselineEntries[key] && candidateEntries[key]
			cohort, order, position := campaignArmCohortPaired, "AB", 1
			if paired && entry.RequestID%2 == 0 {
				order = "BA"
			}
			if paired {
				if (order == "AB" && role == controlledPairRoleCandidate) ||
					(order == "BA" && role == controlledPairRoleBaseline) {
					position = 2
				}
			} else if role == controlledPairRoleBaseline {
				cohort, order = campaignArmCohortBaselineOnly, "A"
			} else {
				cohort, order = campaignArmCohortCandidateOnly, "B"
			}
			blockTime := startedAt.Add(campaignRequestDuration(entry.RequestID))
			observedAt := blockTime
			if position == 2 {
				observedAt = blockTime.Add(20 * time.Millisecond)
			}
			entry.ControlledPair = &controlledPairObservation{
				ContractVersion: controlledPairProtocolVersion,
				SessionID:       sessionID, Protocol: controlledPairInterleaveABBA,
				Role: role, Cohort: cohort, VariantManifestDigest: manifest.ManifestDigest,
				CoordinateDigest: digestString("controlled-pair-coordinate:" + coordinate.canonical()),
				BlockID:          digestString("controlled-pair-test-block:" + coordinate.canonical()),
				Order:            order, Position: position, AttemptID: entry.AttemptID,
				ObservedAt: observedAt, CompletedAt: observedAt.Add(10 * time.Millisecond),
				Load: controlledPairLoadContext{Concurrency: 1, Phase: "single"},
			}
		}
	}
	seal(controlledPairRoleBaseline, fixture.baseline.manifest, fixture.baseline.attestation.Entries)
	seal(controlledPairRoleCandidate, fixture.candidate.manifest, fixture.candidate.attestation.Entries)
	return fixture
}

func nextCampaignRequestID(entries []executionAttestationEntry) uint64 {
	var maximum uint64
	for _, entry := range entries {
		if entry.RequestID > maximum {
			maximum = entry.RequestID
		}
	}
	if maximum == math.MaxUint64 {
		panic("campaign fixture exhausted request identifiers")
	}
	return maximum + 1
}

func campaignRequestDuration(requestID uint64) time.Duration {
	if requestID > maxCampaignTimedRequestID {
		panic("campaign fixture request identifier exceeds the time bound")
	}
	// #nosec G115 -- the explicit bound above proves the value fits time.Duration.
	return time.Duration(requestID) * time.Second
}

func campaignTestPairedGate(id string, evidence CampaignPairedLiveEvidence, fixture campaignPairedLiveFixture) CampaignGate {
	definition, exists := releaseGateDefinitionByID(id)
	if !exists {
		panic("campaign fixture requested an unknown release gate")
	}
	base := CampaignGate{
		ID: id, Name: definition.Name, Disposition: "required", Verdict: "unavailable",
		EvidenceLevel: definition.EvidenceLevel, EvidenceRefs: []string{}, Rationale: "missing",
	}
	return campaignPairedLiveGate(base, evidence, fixture.baseline, fixture.candidate)
}
