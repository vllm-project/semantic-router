package evaluationplane

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestCampaignPairedLiveEvidenceBindsProviderTargetWithoutLiteralTargetAssumptions(t *testing.T) {
	fixture := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	fixture.baseline.report.Run.TargetID = "provider-target-a"
	fixture.baseline.report.Provenance.TargetID = "provider-target-a"
	fixture.baseline.attestation.TargetID = "provider-target-a"
	fixture.candidate.report.Run.TargetID = "provider-target-b"
	fixture.candidate.report.Provenance.TargetID = "provider-target-b"
	fixture.candidate.attestation.TargetID = "provider-target-b"
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil || evidence.BaselineTargetID != "provider-target-a" ||
		evidence.CandidateTargetID != "provider-target-b" || evidence.MixtureID != "mom-core" {
		t.Fatalf("evidence=%+v err=%v", evidence, err)
	}
	fixture.candidate.attestation.TargetID = "provider-target-c"
	if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil {
		t.Fatal("paired evidence with a candidate report/attestation mismatch was accepted")
	}

	sharedTarget := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	sharedTarget.candidate.report.Run.TargetID = sharedTarget.baseline.report.Run.TargetID
	sharedTarget.candidate.report.Provenance.TargetID = sharedTarget.baseline.report.Run.TargetID
	sharedTarget.candidate.attestation.TargetID = sharedTarget.baseline.report.Run.TargetID
	if _, err := buildCampaignPairedLiveEvidence(sharedTarget.baseline, sharedTarget.candidate); err == nil {
		t.Fatal("paired evidence from one deployment target was accepted")
	}

	logicalMismatch := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	candidateMixture := *logicalMismatch.candidate.report.Run.Mixture
	candidateMixture.RecipeName = "different-recipe"
	logicalMismatch.candidate.report.Run.Mixture = &candidateMixture
	if _, err := buildCampaignPairedLiveEvidence(logicalMismatch.baseline, logicalMismatch.candidate); err == nil {
		t.Fatal("paired evidence over different logical Mixture subjects was accepted")
	}
}

func TestGenericComparisonAcceptsOnlyValidatedCrossDeploymentControlledPair(t *testing.T) {
	fixture := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	fixture.baseline.report.Gates = testReleaseGates("recipe", time.Time{})
	fixture.candidate.report.Gates = testReleaseGates("recipe", time.Time{})
	comparison, err := compareControlledPairReports(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatalf("compare controlled pair reports: %v", err)
	}
	if comparison.BaselineRunID != fixture.baseline.report.Run.ID ||
		comparison.CandidateRunID != fixture.candidate.report.Run.ID {
		t.Fatalf("comparison identities = %+v", comparison)
	}
	g3, found := reportGateFromSlice(comparison.Gates, "G3")
	if !found || g3.Verdict != "unavailable" || g3.EvidenceLevel != "E0" ||
		g3.Owner != "recipe-and-model-pool" || g3.Observed != nil || g3.Threshold != nil ||
		g3.SampleCount != nil || len(g3.EvidenceRefs) != 4 ||
		g3.EvidenceRefs[0] != comparativeG3ReductionRef {
		t.Fatalf("generic controlled-pair G3 must remain an E0 diagnostic: %+v", g3)
	}

	fixture.candidate.attestation.Entries[0].ControlledPair = nil
	if _, err := compareControlledPairReports(fixture.baseline, fixture.candidate); err == nil {
		t.Fatal("comparison accepted cross-deployment reports without controlled-pair provenance")
	}
}

func TestCrossDeploymentComparisonRejectsInvalidControlledPairAsDomainError(t *testing.T) {
	unbound := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	unbound.candidate.manifest.BaselineRunID = ""
	if _, err := compareControlledPairReports(unbound.baseline, unbound.candidate); !errors.Is(err, ErrInvalid) ||
		!strings.Contains(err.Error(), "server-owned controlled pair") {
		t.Fatalf("unbound controlled pair error=%v, want actionable ErrInvalid", err)
	}

	topologyChanged := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
	topologyChanged.candidate.manifest.Target.BackendTopologyDigest = digestString("different-topology")
	if _, err := compareControlledPairReports(topologyChanged.baseline, topologyChanged.candidate); !errors.Is(err, ErrInvalid) ||
		!strings.Contains(err.Error(), "backend topology changed outside the declared treatment") {
		t.Fatalf("undeclared topology change error=%v, want actionable ErrInvalid", err)
	}
}
