package evaluationplane

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

type controlledPairAdmission struct {
	observation *controlledPairObservation
	lease       *controlledPairLease
	err         error
}

func TestControlledPairCoordinatorAdmitsOnlyABBAOrderedCounterparts(t *testing.T) {
	baseline := controlledPairTestManifest("baseline-manifest", "http://baseline.test")
	candidate := controlledPairTestManifest("candidate-manifest", "http://candidate.test")
	coordinator := newControlledPairCoordinator(
		"b15a6e5e-653a-438b-aa14-cbe03dbf9a41", 19, baseline, candidate,
	)
	request := workerBrokerRequest{
		ID: 1, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
		CaseID: "case-1", AttemptID: "attempt-case-1", TimeoutMS: 1000,
	}
	payload, err := json.Marshal(map[string]any{"model": "vllm-sr/auto", "messages": []any{}})
	if err != nil {
		t.Fatal(err)
	}
	admissions := make(chan controlledPairAdmission, 2)
	for _, role := range []string{controlledPairRoleBaseline, controlledPairRoleCandidate} {
		go func() {
			observation, lease, admissionErr := coordinator.before(context.Background(), role, request, payload)
			admissions <- controlledPairAdmission{observation: observation, lease: lease, err: admissionErr}
		}()
	}
	first := <-admissions
	if first.err != nil || first.observation == nil || first.observation.Position != 1 {
		t.Fatalf("first admission=%+v", first)
	}
	select {
	case second := <-admissions:
		t.Fatalf("second position admitted before first completed: %+v", second)
	case <-time.After(20 * time.Millisecond):
	}
	firstCompletedAt := time.Now().UTC()
	first.lease.complete(firstCompletedAt)
	second := <-admissions
	if second.err != nil || second.observation == nil || second.observation.Position != 2 ||
		second.observation.SessionID != first.observation.SessionID ||
		second.observation.BlockID != first.observation.BlockID ||
		second.observation.Order != first.observation.Order ||
		second.observation.Role == first.observation.Role ||
		second.observation.ObservedAt.Before(firstCompletedAt) {
		t.Fatalf("second admission=%+v first=%+v", second, first)
	}
	second.lease.complete(time.Now().UTC())
}

func TestControlledPairAddressabilityRejectsOneEndpointPretendingToBeTwoVersions(t *testing.T) {
	baseline := controlledPairTestManifest("baseline-manifest", "http://one-version.test")
	candidate := controlledPairTestManifest("candidate-manifest", "http://one-version.test")
	if err := validateControlledPairAddressability(baseline, candidate); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned") {
		t.Fatalf("same endpoint accepted as two versions: %v", err)
	}
}

func TestCampaignRejectsPostHocPairingOfIndependentLiveRuns(t *testing.T) {
	fixture := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	for index := range fixture.baseline.attestation.Entries {
		fixture.baseline.attestation.Entries[index].ControlledPair = nil
	}
	for index := range fixture.candidate.attestation.Entries {
		fixture.candidate.attestation.Entries[index].ControlledPair = nil
	}
	if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil {
		t.Fatalf("independent runs produced causal paired evidence: %v", err)
	}
}

func TestCampaignRejectsPairBlockWithoutCompletionBeforeCounterpartAdmission(t *testing.T) {
	fixture := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	baselinePair := fixture.baseline.attestation.Entries[0].ControlledPair
	candidatePair := fixture.candidate.attestation.Entries[0].ControlledPair
	first, second := baselinePair, candidatePair
	if candidatePair.Position == 1 {
		first, second = candidatePair, baselinePair
	}
	first.CompletedAt = second.ObservedAt.Add(time.Nanosecond)
	if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil ||
		!strings.Contains(err.Error(), "timestamps") {
		t.Fatalf("overlapping pair block accepted: %v", err)
	}
}

func controlledPairTestManifest(label, origin string) RunManifest {
	return RunManifest{
		ManifestDigest: digestString(label), Mode: ModeLive, Concurrency: 4,
		TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		Target: ManifestTarget{
			RouterAPIURL: origin, EnvoyURL: origin,
			Mixture: &ManifestMixture{
				ID: "mom-core", ModelArms: []ModelArm{
					{ID: "arm-a", Model: "model-a"}, {ID: "arm-b", Model: "model-b"},
				},
			},
		},
	}
}
