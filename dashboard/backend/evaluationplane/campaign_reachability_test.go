package evaluationplane

import (
	"strings"
	"testing"
)

func TestEveryRequiredCampaignSlotHasAConstructibleBinding(t *testing.T) {
	for _, profile := range builtinChangeProfiles() {
		t.Run(string(profile.ID), func(t *testing.T) {
			assertRequiredCampaignSlotsReachable(t, profile)
		})
	}
}

func assertRequiredCampaignSlotsReachable(t *testing.T, profile CatalogChangeProfile) {
	t.Helper()
	subject := digestString("reachable-candidate:" + string(profile.ID))
	for index, slot := range profile.CampaignSlots {
		if slot.Disposition != "required" {
			continue
		}
		t.Run(slot.GateID, func(t *testing.T) {
			assertCampaignSlotReachable(t, profile.ID, slot, index, subject)
		})
	}
}

func assertCampaignSlotReachable(
	t *testing.T,
	profile ChangeProfile,
	slot CatalogCampaignSlot,
	index int,
	subject string,
) {
	t.Helper()
	switch slot.BindingKind {
	case CampaignBindingRun:
		assertCampaignRunSlotReachable(t, profile, slot, index, subject)
	case CampaignBindingControlledPair:
		assertCampaignControlledPairReachable(t, profile, slot)
	case CampaignBindingFidelityPair:
		assertCampaignFidelityPairReachable(t, profile, slot, subject)
	default:
		t.Fatalf("required slot has unknown binding kind %q", slot.BindingKind)
	}
}

func assertCampaignRunSlotReachable(
	t *testing.T,
	profile ChangeProfile,
	slot CatalogCampaignSlot,
	index int,
	subject string,
) {
	t.Helper()
	runID := campaignReachabilityRunID(index)
	evidence := campaignV2SingleRunEvidence(profile, slot.GateID, runID, subject)
	binding := campaignEvidenceBinding{
		slotID: strings.ToLower(slot.GateID), gateID: slot.GateID,
		bindingRole: campaignSingleBindingRole, runID: runID, candidate: true,
	}
	if err := validateCampaignBoundRun(profile, binding, slot, evidence); err != nil {
		t.Fatalf("required run slot is unreachable: %v", err)
	}
}

func assertCampaignControlledPairReachable(
	t *testing.T,
	profile ChangeProfile,
	slot CatalogCampaignSlot,
) {
	t.Helper()
	fixture := reachableCampaignG3Fixture(profile)
	for _, item := range []struct {
		role     string
		evidence campaignRunEvidence
	}{
		{role: "baseline", evidence: fixture.baseline},
		{role: "candidate", evidence: fixture.candidate},
	} {
		binding := campaignEvidenceBinding{
			slotID: "g3", gateID: "G3", bindingRole: item.role,
			runID: item.evidence.report.Run.ID, candidate: item.role == "candidate",
		}
		if err := validateCampaignBoundRun(profile, binding, slot, item.evidence); err != nil {
			t.Fatalf("required controlled-pair %s is unreachable: %v", item.role, err)
		}
	}
	if err := validateCampaignPairedLiveSources(fixture.baseline, fixture.candidate); err != nil {
		t.Fatalf("required controlled pair is unreachable: %v", err)
	}
}

func assertCampaignFidelityPairReachable(
	t *testing.T,
	profile ChangeProfile,
	slot CatalogCampaignSlot,
	subject string,
) {
	t.Helper()
	reference, live := reachableCampaignG5Fixture(t, profile, subject)
	for _, item := range []struct {
		role     string
		evidence campaignRunEvidence
	}{
		{role: "reference", evidence: reference},
		{role: "live", evidence: live},
	} {
		binding := campaignEvidenceBinding{
			slotID: "g5", gateID: "G5", bindingRole: item.role,
			runID: item.evidence.report.Run.ID, candidate: true,
		}
		if err := validateCampaignBoundRun(profile, binding, slot, item.evidence); err != nil {
			t.Fatalf("required fidelity %s is unreachable: %v", item.role, err)
		}
	}
	evidence, err := buildCampaignFidelityEvidence(reference, live)
	if err != nil || evidence.TrackID != slot.TrackID {
		t.Fatalf("required fidelity pair is unreachable: evidence=%+v err=%v", evidence, err)
	}
}

func TestCampaignCatalogRejectsRequiredG3WithoutPairedTreatment(t *testing.T) {
	registry := emptyRegistry()
	for _, executor := range builtinExecutorContracts() {
		registry.executors[executor.ID] = executor
	}
	for _, track := range builtinTracks() {
		registry.tracks[track.ID] = track
	}
	profiles := builtinChangeProfiles()
	for profileIndex := range profiles {
		if profiles[profileIndex].ID != "agent_multimodal" {
			continue
		}
		for slotIndex := range profiles[profileIndex].CampaignSlots {
			if profiles[profileIndex].CampaignSlots[slotIndex].GateID == "G3" {
				profiles[profileIndex].CampaignSlots[slotIndex].Disposition = "required"
			}
		}
	}
	if err := validateCampaignCatalogContracts(registry, profiles); err == nil ||
		!strings.Contains(err.Error(), "without a paired treatment protocol") {
		t.Fatalf("structurally unreachable G3 contract error=%v", err)
	}
}

func TestControlledPairReachabilityFailsClosedForCatalogSingleOriginsAndRouterAuth(t *testing.T) {
	fixture := reachableCampaignG3Fixture("recipe")
	fixture.candidate.manifest.Target.RouterAPIURL = fixture.baseline.manifest.Target.RouterAPIURL
	fixture.candidate.manifest.Target.EnvoyURL = fixture.baseline.manifest.Target.EnvoyURL
	if err := validateControlledPairAddressability(fixture.baseline.manifest, fixture.candidate.manifest); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned Router origins") {
		t.Fatalf("single-origin controlled pair error=%v", err)
	}

	arms := []ModelArm{
		catalogTestArm("text-a", []string{"text"}),
		catalogTestArm("text-b", []string{"text"}),
	}
	registry, err := NewRegistry(
		"http://router.test", "http://envoy.test",
		RegistryOptions{
			Mixtures:           []MixtureTargetSnapshot{catalogTestMixtureSnapshot(arms, catalogTopologyDigest)},
			RouterAuthRequired: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range registry.targets {
		if target.Public.Kind != "mixture-of-models" {
			continue
		}
		if target.RouterAPIURL != "" || containsTrack(target.Public.TrackIDs, "routing") {
			t.Fatalf("auth-blocked Router was advertised as a reachable routing target: %+v", target)
		}
	}
}

func campaignReachabilityRunID(index int) string {
	return []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
		"33333333-3333-4333-8333-333333333333",
		"44444444-4444-4444-8444-444444444444",
		"55555555-5555-4555-8555-555555555555",
		"66666666-6666-4666-8666-666666666666",
		"77777777-7777-4777-8777-777777777777",
		"88888888-8888-4888-8888-888888888888",
	}[index]
}

func reachableCampaignG3Fixture(profile ChangeProfile) campaignPairedLiveFixture {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	if profile == "model_pool" {
		fixture = withChangedCandidatePool(fixture)
	}
	prepareCampaignV2G3Run(&fixture.baseline, profile, "")
	prepareCampaignV2G3Run(&fixture.candidate, profile, digestString("reachable-g3-candidate:"+string(profile)))

	switch profile {
	case "selector":
		freezeCampaignPolicy(&fixture)
		fixture.candidate.report.Run.Mixture.SelectorDigest = digestString("reachable-selector-treatment")
	case "online_adaptation":
		freezeCampaignPolicy(&fixture)
		fixture.candidate.report.Run.Mixture.AdaptationDigest = digestString("reachable-adaptation-treatment")
	}
	return fixture
}

func freezeCampaignPolicy(fixture *campaignPairedLiveFixture) {
	fixture.candidate.report.Provenance.PolicySnapshotDigest = fixture.baseline.report.Provenance.PolicySnapshotDigest
	fixture.candidate.report.Run.Mixture.RecipeDigest = fixture.baseline.report.Run.Mixture.RecipeDigest
	fixture.candidate.attestation.PolicySnapshotDigest = fixture.baseline.attestation.PolicySnapshotDigest
}

func reachableCampaignG5Fixture(
	t *testing.T,
	profile ChangeProfile,
	subject string,
) (campaignRunEvidence, campaignRunEvidence) {
	t.Helper()
	if profile == "agent_multimodal" {
		return campaignAgentMultimodalFidelityFixture(t)
	}
	reference, live := campaignFidelityV2Fixture(t)
	prepareCampaignV2G5Run(&reference, profile, subject)
	prepareCampaignV2G5Run(&live, profile, subject)
	return reference, live
}
