package evaluationplane

import (
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestDeletedControlledPairPermanentlyReservesMemberRunIdentities(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	owner := testLifecycleActor(t, "pair-reservation-owner", false)
	other := testLifecycleActor(t, "pair-reservation-other", false)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, owner)
	if _, err := service.store.createControlledPairBundlesAs(
		owner, pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish reservation fixture pair: %v", err)
	}
	if err := service.DeleteControlledPairExecutionAs(owner, pair.PairID); err != nil {
		t.Fatalf("delete reservation fixture pair: %v", err)
	}

	tombstoneBytes, tombstoneErr := privateDirectoryBytes(
		filepath.Join(service.store.controlledPairRoot, pair.PairID), "",
	)
	if tombstoneErr != nil || tombstoneBytes == 0 {
		t.Fatalf("measure controlled pair tombstone: bytes=%d err=%v", tombstoneBytes, tombstoneErr)
	}
	beforeReport, before := controlledPairReservationUsage(t, service.store, owner)
	if before.ActualBytes < tombstoneBytes || before.ChargeableBytes > before.MaxBytes ||
		beforeReport.ChargeableBytes > beforeReport.MaxStoreBytes {
		t.Fatalf("tombstone is not bounded usage: report=%+v owner=%+v tombstone=%d", beforeReport, before, tombstoneBytes)
	}

	reopened := newTestPeerStore(t, service.store)
	afterReport, after := controlledPairReservationUsage(t, reopened, owner)
	if after.ActualBytes < tombstoneBytes || after.ChargeableBytes > after.MaxBytes ||
		afterReport.ChargeableBytes > afterReport.MaxStoreBytes {
		t.Fatalf("reopened tombstone is not bounded usage: report=%+v owner=%+v tombstone=%d", afterReport, after, tombstoneBytes)
	}
	service.store = reopened

	for role, reservedID := range map[string]string{
		"baseline": pair.BaselineRunID, "candidate": pair.CandidateRunID,
	} {
		t.Run("ordinary_"+role, func(t *testing.T) {
			for name, attempt := range map[string]struct {
				actor Actor
				want  error
			}{
				"owner": {actor: owner, want: ErrConflict},
				"other": {actor: other, want: ErrForbidden},
			} {
				t.Run(name, func(t *testing.T) {
					run, manifest := ordinaryRunForReservedIdentity(t, pair.BaselineRun, baselineManifest, reservedID)
					if _, err := reopened.CreateBundleAs(attempt.actor, run, manifest); !errors.Is(err, attempt.want) {
						t.Fatalf("reserved ordinary run error=%v, want %v", err, attempt.want)
					}
				})
			}
		})
	}

	ownerBaselineSource, err := service.loadControlledPairSource(pair.BaselineSourceRunID)
	if err != nil {
		t.Fatalf("reload owner baseline source: %v", err)
	}
	ownerCandidateSource, err := service.loadControlledPairSource(pair.CandidateSourceRunID)
	if err != nil {
		t.Fatalf("reload owner candidate source: %v", err)
	}
	otherBaselineSource := createStoreControlledPairSource(t, service, other, pair.BaselineRun.TargetID)
	otherCandidateSource := createStoreControlledPairSource(t, service, other, pair.CandidateRun.TargetID)
	for role, reservedID := range map[string]string{
		"baseline": pair.BaselineRunID, "candidate": pair.CandidateRunID,
	} {
		t.Run("new_pair_"+role, func(t *testing.T) {
			for name, attempt := range map[string]struct {
				actor               Actor
				baseline, candidate controlledPairSource
				want                error
			}{
				"owner": {
					actor: owner, baseline: ownerBaselineSource, candidate: ownerCandidateSource,
					want: ErrConflict,
				},
				"other": {
					actor: other, baseline: otherBaselineSource, candidate: otherCandidateSource,
					want: ErrForbidden,
				},
			} {
				t.Run(name, func(t *testing.T) {
					candidatePair, baseline, candidate := controlledPairWithReservedMember(
						t, attempt.actor, attempt.baseline, attempt.candidate, role, reservedID,
					)
					if _, err := reopened.createControlledPairBundlesAs(
						attempt.actor, candidatePair, baseline, candidate,
					); !errors.Is(err, attempt.want) {
						t.Fatalf("reserved controlled pair member error=%v, want %v", err, attempt.want)
					}
				})
			}
		})
	}
}

func controlledPairReservationUsage(
	t *testing.T,
	store *Store,
	owner Actor,
) (LifecycleUsageReport, OwnerLifecycleUsage) {
	t.Helper()
	report, err := store.Usage(owner)
	if err != nil || len(report.Owners) != 1 {
		t.Fatalf("read controlled pair reservation usage: owners=%d err=%v", len(report.Owners), err)
	}
	return report, report.Owners[0]
}

func ordinaryRunForReservedIdentity(
	t *testing.T,
	template Run,
	templateManifest RunManifest,
	runID string,
) (Run, RunManifest) {
	t.Helper()
	run := template
	run.ID, run.ClientRequestID, run.BaselineRunID, run.ControlledPair = runID, runID, "", nil
	manifest := templateManifest
	manifest.RunID, manifest.BaselineRunID = runID, ""
	refreshTestManifestDigest(t, &manifest)
	return run, manifest
}

func controlledPairWithReservedMember(
	t *testing.T,
	actor Actor,
	baselineSource controlledPairSource,
	candidateSource controlledPairSource,
	role string,
	reservedID string,
) (controlledPairManifest, RunManifest, RunManifest) {
	t.Helper()
	baselineID, candidateID := newTestClientRequestID(), newTestClientRequestID()
	if role == "baseline" {
		baselineID = reservedID
	} else {
		candidateID = reservedID
	}
	createdAt := time.Now().UTC().Truncate(time.Microsecond)
	baseline, baselineManifest, err := cloneControlledPairRun(
		baselineSource, baselineID, "", controlledPairRoleBaseline, createdAt,
	)
	if err != nil {
		t.Fatalf("clone reservation baseline: %v", err)
	}
	candidate, candidateManifest, err := cloneControlledPairRun(
		candidateSource, candidateID, baselineID, controlledPairRoleCandidate, createdAt.Add(time.Microsecond),
	)
	if err != nil {
		t.Fatalf("clone reservation candidate: %v", err)
	}
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.run.ID,
		CandidateSourceRunID: candidateSource.run.ID, BaselineRunID: baselineID, CandidateRunID: candidateID,
	}
	pair, err := newControlledPairManifest(
		actor, request, baselineSource, candidateSource,
		baseline, candidate, baselineManifest, candidateManifest,
	)
	if err != nil {
		t.Fatalf("build reservation pair: %v", err)
	}
	return pair, baselineManifest, candidateManifest
}
