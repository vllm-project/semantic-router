package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestCampaignReadinessRequestIsBoundedAndCanonical(t *testing.T) {
	validID := "11111111-1111-4111-8111-111111111111"
	if err := validateCampaignReadinessRequest(CampaignReadinessRequest{
		ChangeProfile: "recipe", Limit: maxRunPageLimit,
		BaselineSourceRunID: validID, ReferenceRunID: "22222222-2222-4222-8222-222222222222",
	}); err != nil {
		t.Fatalf("valid readiness request: %v", err)
	}
	for _, test := range []CampaignReadinessRequest{
		{ChangeProfile: "unknown"},
		{ChangeProfile: "recipe", Limit: -1},
		{ChangeProfile: "recipe", Limit: maxRunPageLimit + 1},
		{ChangeProfile: "recipe", BaselineSourceRunID: "not-a-run"},
		{ChangeProfile: "recipe", ReferenceRunID: "not-a-run"},
		{ChangeProfile: "recipe", Cursor: string(make([]byte, maxRunListCursorLength+1))},
	} {
		if err := validateCampaignReadinessRequest(test); !errors.Is(err, ErrInvalid) {
			t.Fatalf("request=%+v error=%v, want ErrInvalid", test, err)
		}
	}
}

func TestEmptyCampaignReadinessPreservesCanonicalSlotShape(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	readiness, err := service.CampaignReadinessAs(SystemActor(), CampaignReadinessRequest{
		ChangeProfile: "recipe",
	})
	if err != nil {
		t.Fatalf("CampaignReadinessAs: %v", err)
	}
	profile, _ := campaignProfileContract("recipe")
	if readiness.SchemaVersion != SchemaVersion || readiness.ChangeProfile != "recipe" ||
		len(readiness.Slots) != len(profile.CampaignSlots) {
		t.Fatalf("readiness=%+v", readiness)
	}
	for index, slot := range readiness.Slots {
		if slot.GateID != profile.CampaignSlots[index].GateID ||
			slot.BindingKind != profile.CampaignSlots[index].BindingKind ||
			slot.EligibleRunIDs == nil || slot.ControlledPairSourceRunIDs == nil ||
			slot.ControlledPairCandidateRunIDs == nil || slot.FidelityReferenceRunIDs == nil ||
			slot.FidelityLiveRunIDs == nil {
			t.Fatalf("slot %d=%+v", index, slot)
		}
	}
}

func TestCampaignReadinessEvidenceLeaseDoesNotHoldLifecycleLock(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRunAs: %v", err)
	}
	_, _, _, releaseEvidence, err := service.prepareCampaignReadiness(SystemActor(), CampaignReadinessRequest{
		ChangeProfile: "recipe", BaselineSourceRunID: run.ID,
	})
	if err != nil {
		t.Fatalf("prepareCampaignReadiness: %v", err)
	}
	defer releaseEvidence()

	done := make(chan error, 1)
	go func() {
		_, readErr := service.GetRunAs(SystemActor(), run.ID)
		done <- readErr
	}()
	select {
	case readErr := <-done:
		if readErr != nil {
			t.Fatalf("unrelated lifecycle read: %v", readErr)
		}
	case <-time.After(time.Second):
		t.Fatal("readiness evidence lease retained the lifecycle lock")
	}
}

func TestCampaignReadinessProjectsOnlyCanonicalRunAndFidelityBindings(t *testing.T) {
	subject := digestString("readiness-subject")
	valid := campaignV2SingleRunEvidence(
		"schema_adapter", "G4", "44444444-4444-4444-8444-444444444444", subject,
	)
	invalid := campaignV2SingleRunEvidence(
		"schema_adapter", "G4", "55555555-5555-4555-8555-555555555555", subject,
	)
	invalid.report.Gates = nil
	slot, _ := campaignSlotContract("schema_adapter", "G4")
	runs := []Run{valid.report.Run, invalid.report.Run}
	ready := campaignReadyRunIDs("schema_adapter", slot, runs, map[string]campaignRunEvidence{
		valid.report.Run.ID: valid, invalid.report.Run.ID: invalid,
	})
	if len(ready) != 1 || ready[0] != valid.report.Run.ID {
		t.Fatalf("ready run IDs=%v", ready)
	}

	reference, live := reachableCampaignG5Fixture(t, "recipe", subject)
	fidelitySlot, _ := campaignSlotContract("recipe", "G5")
	references, liveRuns := campaignReadyFidelitySlot(
		"recipe", fidelitySlot, []Run{reference.report.Run, live.report.Run},
		[]Run{reference.report.Run, live.report.Run}, &reference.report.Run,
		map[string]campaignRunEvidence{
			reference.report.Run.ID: reference,
			live.report.Run.ID:      live,
		},
	)
	if !campaignStringMember(references, reference.report.Run.ID) ||
		len(liveRuns) != 1 || liveRuns[0] != live.report.Run.ID {
		t.Fatalf("fidelity readiness references=%v live=%v", references, liveRuns)
	}

	scans := 0
	_, linearLive := campaignReadyFidelitySlotWithScanner(
		"recipe", fidelitySlot, []Run{reference.report.Run, live.report.Run},
		[]Run{reference.report.Run, live.report.Run}, &reference.report.Run,
		map[string]campaignRunEvidence{
			reference.report.Run.ID: reference,
			live.report.Run.ID:      live,
		},
		func(campaignRunEvidence, TrackID) (string, error) {
			scans++
			return digestString("one-readiness-record-cohort"), nil
		},
	)
	if scans != 2 || len(linearLive) != 1 {
		t.Fatalf("fidelity readiness scanned records %d times for 2 runs; live=%v", scans, linearLive)
	}
}

func TestCampaignReadinessCarriesPaginationBeyondOneMaximumPage(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	runs := make([]Run, 0, maxRunPageLimit+1)
	ownerDigests := make(map[string]string, maxRunPageLimit+1)
	presentRunIDs := make(map[string]bool, maxRunPageLimit+1)
	createdAt := time.Date(2026, time.September, 2, 1, 2, 3, 0, time.UTC)
	for index := 0; index <= maxRunPageLimit; index++ {
		runID := fmt.Sprintf("10000000-0000-4000-8000-%012d", index)
		runs = append(runs, Run{ID: runID, CreatedAt: createdAt.Add(time.Duration(index) * time.Second)})
		ownerDigests[runID] = SystemActor().principalDigest
		presentRunIDs[runID] = true
	}
	service.store.runIndex.replace(
		runs, ownerDigests, presentRunIDs, map[string]runListWarning{}, 0, map[string]int{},
	)

	first, err := service.CampaignReadinessAs(SystemActor(), CampaignReadinessRequest{
		ChangeProfile: "recipe", Limit: maxRunPageLimit,
	})
	if err != nil {
		t.Fatalf("first readiness page: %v", err)
	}
	if first.TotalRuns != maxRunPageLimit+1 || first.NextCursor == "" {
		t.Fatalf("first paged readiness=%+v", first)
	}
	second, err := service.CampaignReadinessAs(SystemActor(), CampaignReadinessRequest{
		ChangeProfile: "recipe", Limit: maxRunPageLimit, Cursor: first.NextCursor,
	})
	if err != nil {
		t.Fatalf("second readiness page: %v", err)
	}
	if second.TotalRuns != maxRunPageLimit+1 || second.NextCursor != "" {
		t.Fatalf("second paged readiness=%+v", second)
	}
}
