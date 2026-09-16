package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

type failingLifecyclePersistence struct{}

func (failingLifecyclePersistence) Write(string, any) error {
	return errors.New("injected lifecycle publication failure")
}

func (failingLifecyclePersistence) SyncDirectory(string, string) error { return nil }

type failingLifecycleAuditWriter struct{}

func (failingLifecycleAuditWriter) WriteExclusive(string, any) error {
	return errors.New("injected lifecycle audit failure")
}

func (failingLifecycleAuditWriter) SyncDirectory(string, string) error { return nil }

func TestLifecycleOwnerAuthorizationAndHashChainedAudit(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "owner-a", false)
	other := testLifecycleActor(t, "owner-b", false)
	administrator := testLifecycleActor(t, "admin", true)

	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRunAs owner: %v", err)
	}
	if _, err := service.CreateRunAs(context.Background(), other, requestForExistingRun(run)); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner idempotent create error=%v, want ErrForbidden", err)
	}
	hold := true
	protected := RetentionProtected
	if _, err := service.UpdateRunLifecycle(other, run.ID, UpdateLifecycleRequest{
		RetentionClass: &protected, EvidenceHold: &hold,
	}); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner retention/hold error=%v, want ErrForbidden", err)
	}
	hold = false
	if _, err := service.UpdateRunLifecycle(other, run.ID, UpdateLifecycleRequest{
		EvidenceHold: &hold,
	}); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner release error=%v, want ErrForbidden", err)
	}
	hold = true
	if _, err := service.UpdateRunLifecycle(owner, run.ID, UpdateLifecycleRequest{
		RetentionClass: &protected, EvidenceHold: &hold,
	}); err != nil {
		t.Fatalf("owner retention/hold: %v", err)
	}
	hold = false
	standard := RetentionStandard
	if _, err := service.UpdateRunLifecycle(owner, run.ID, UpdateLifecycleRequest{
		RetentionClass: &standard, EvidenceHold: &hold,
	}); err != nil {
		t.Fatalf("owner retention/release: %v", err)
	}
	if _, err := service.CollectLifecycle(other, CollectionRequest{}); !errors.Is(err, ErrForbidden) {
		t.Fatalf("non-administrator collection error=%v, want ErrForbidden", err)
	}
	if err := service.DeleteRunAs(other, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner delete error=%v, want ErrForbidden", err)
	}
	if err := service.DeleteRunAs(administrator, run.ID); err != nil {
		t.Fatalf("administrator delete: %v", err)
	}

	records := lifecycleAuditRecords(service.store)
	assertLifecycleAuditDecision(t, records, "create", "allowed")
	assertLifecycleAuditDecision(t, records, "create", "denied")
	assertLifecycleAuditDecision(t, records, "retention", "denied")
	assertLifecycleAuditDecision(t, records, "retention", "allowed")
	assertLifecycleAuditDecision(t, records, "hold", "denied")
	assertLifecycleAuditDecision(t, records, "hold", "allowed")
	assertLifecycleAuditDecision(t, records, "release", "denied")
	assertLifecycleAuditDecision(t, records, "release", "allowed")
	assertLifecycleAuditDecision(t, records, "delete", "denied")
	assertLifecycleAuditDecision(t, records, "delete", "allowed")
	assertLifecycleAuditDecision(t, records, "gc", "denied")
	assertLifecycleAuditDecision(t, records, "gc", "allowed")
	if err := service.Close(); err != nil {
		t.Fatalf("close before lifecycle audit restart: %v", err)
	}
	if _, err := newStandaloneStore(service.store.root); err != nil {
		t.Fatalf("restart with valid lifecycle audit: %v", err)
	}
}

func TestLifecycleStartAndCancelRequireOwnerOrAdministrator(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 1)}
	service, _ := newTestService(t, process, 1)
	owner := testLifecycleActor(t, "owner", false)
	other := testLifecycleActor(t, "other", false)
	administrator := testLifecycleActor(t, "admin", true)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRunAs: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), other, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner start error=%v, want ErrForbidden", err)
	}
	if _, err := service.StartRunAs(context.Background(), owner, run.ID); err != nil {
		t.Fatalf("owner start: %v", err)
	}
	<-process.started
	if _, err := service.CancelRunAs(other, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner cancel error=%v, want ErrForbidden", err)
	}
	if _, err := service.CancelRunAs(administrator, run.ID); err != nil {
		t.Fatalf("administrator cancel: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	records := lifecycleAuditRecords(service.store)
	assertLifecycleAuditDecision(t, records, "start", "denied")
	assertLifecycleAuditDecision(t, records, "start", "allowed")
	assertLifecycleAuditDecision(t, records, "cancel", "denied")
	assertLifecycleAuditDecision(t, records, "cancel", "allowed")
}

func TestHeldProtectedAndReferencedRunsCannotBeCollected(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "owner", false)
	administrator := testLifecycleActor(t, "admin", true)
	baseline, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)

	hold := true
	if _, holdErr := service.UpdateRunLifecycle(owner, baseline.ID, UpdateLifecycleRequest{EvidenceHold: &hold}); holdErr != nil {
		t.Fatalf("hold baseline: %v", holdErr)
	}
	if deleteHeldErr := service.DeleteRunAs(administrator, baseline.ID); !errors.Is(deleteHeldErr, ErrConflict) {
		t.Fatalf("delete held run error=%v, want ErrConflict", deleteHeldErr)
	}
	hold = false
	protected := RetentionProtected
	if _, protectErr := service.UpdateRunLifecycle(owner, baseline.ID, UpdateLifecycleRequest{
		EvidenceHold: &hold, RetentionClass: &protected,
	}); protectErr != nil {
		t.Fatalf("release and protect baseline: %v", protectErr)
	}
	if deleteProtectedErr := service.DeleteRunAs(administrator, baseline.ID); !errors.Is(deleteProtectedErr, ErrConflict) {
		t.Fatalf("delete protected run error=%v, want ErrConflict", deleteProtectedErr)
	}
	standard := RetentionStandard
	if _, restoreRetentionErr := service.UpdateRunLifecycle(owner, baseline.ID, UpdateLifecycleRequest{RetentionClass: &standard}); restoreRetentionErr != nil {
		t.Fatalf("restore standard retention: %v", restoreRetentionErr)
	}

	candidateRequest := validCreateRequest()
	candidateRequest.BaselineRunID = baseline.ID
	candidate, manifest := preparePendingRun(t, service, candidateRequest)
	if _, publishCandidateErr := service.store.CreateBundleAs(owner, candidate, manifest); publishCandidateErr != nil {
		t.Fatalf("publish referenced candidate: %v", publishCandidateErr)
	}
	candidate = completeTestRun(t, service, candidate)
	service.store.lifecycleNow = func() time.Time { return time.Now().UTC().Add(90 * 24 * time.Hour) }
	hold = true
	if _, holdCandidateErr := service.UpdateRunLifecycle(owner, candidate.ID, UpdateLifecycleRequest{EvidenceHold: &hold}); holdCandidateErr != nil {
		t.Fatalf("hold already-expired candidate: %v", holdCandidateErr)
	}
	heldPlan, err := service.CollectLifecycle(administrator, CollectionRequest{})
	if err != nil || len(heldPlan.Plan.Candidates) != 0 || heldPlan.Plan.Skipped["held"] != 1 {
		t.Fatalf("expired evidence hold was not collection-safe: plan=%+v err=%v", heldPlan.Plan, err)
	}
	hold = false
	if _, releaseCandidateErr := service.UpdateRunLifecycle(owner, candidate.ID, UpdateLifecycleRequest{EvidenceHold: &hold}); releaseCandidateErr != nil {
		t.Fatalf("release already-expired candidate: %v", releaseCandidateErr)
	}

	dryRun, err := service.CollectLifecycle(administrator, CollectionRequest{})
	if err != nil {
		t.Fatalf("dry-run collection: %v", err)
	}
	if len(dryRun.Plan.Candidates) != 1 || dryRun.Plan.Candidates[0].RunID != candidate.ID ||
		dryRun.Plan.Skipped["referenced"] != 1 {
		t.Fatalf("collection plan did not preserve referenced baseline: %+v", dryRun.Plan)
	}
	if _, stalePlanErr := service.CollectLifecycle(administrator, CollectionRequest{Apply: true, PlanDigest: digestString("stale")}); !errors.Is(stalePlanErr, ErrConflict) {
		t.Fatalf("stale collection error=%v, want ErrConflict", stalePlanErr)
	}
	applied, err := service.CollectLifecycle(administrator, CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest})
	if err != nil {
		t.Fatalf("apply collection: %v", err)
	}
	if !applied.Applied || !reflect.DeepEqual(applied.DeletedRunIDs, []string{candidate.ID}) {
		t.Fatalf("applied collection=%+v", applied)
	}
	if _, err := service.GetRunAs(SystemActor(), baseline.ID); err != nil {
		t.Fatalf("referenced baseline was deleted: %v", err)
	}
}

func TestCancelledRunParticipatesInRetentionCollection(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "owner", false)
	administrator := testLifecycleActor(t, "admin", true)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create run: %v", err)
	}
	now := time.Now().UTC()
	run.Status, run.CompletedAt = StatusCancelled, &now
	run.Progress.Message = "Run cancelled"
	if persistCancelledErr := service.store.updateRunFixture(run); persistCancelledErr != nil {
		t.Fatalf("persist cancelled run: %v", persistCancelledErr)
	}
	service.store.lifecycleNow = func() time.Time { return now.Add(90 * 24 * time.Hour) }
	dryRun, err := service.CollectLifecycle(administrator, CollectionRequest{})
	if err != nil {
		t.Fatalf("plan cancelled run collection: %v", err)
	}
	if len(dryRun.Plan.Candidates) != 1 || dryRun.Plan.Candidates[0].RunID != run.ID {
		t.Fatalf("cancelled run was omitted from retention collection: %+v", dryRun.Plan)
	}
	if _, err := service.CollectLifecycle(administrator, CollectionRequest{
		Apply: true, PlanDigest: dryRun.Plan.PlanDigest,
	}); err != nil {
		t.Fatalf("collect cancelled run: %v", err)
	}
	if _, err := service.GetRunAs(SystemActor(), run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cancelled run remains after collection: %v", err)
	}
}

func TestLifecycleQuotasUsageAndRepeatedGrowthRemainBounded(t *testing.T) {
	limits := LifecycleLimits{
		MaxOwnerBytes:     25 * 1024 * 1024,
		MaxStoreBytes:     55*1024*1024 + lifecycleCollectionReservedBytes,
		MaxOwnerRuns:      1,
		MaxOwnerCampaigns: 1,
		MaxAuditBytes:     2 * 1024 * 1024,
	}
	service, _ := newLifecycleTestService(t, limits)
	ownerA := testLifecycleActor(t, "owner-a", false)
	ownerB := testLifecycleActor(t, "owner-b", false)
	ownerC := testLifecycleActor(t, "owner-c", false)
	attestedRequest := validCreateRequest()
	attestedRequest.Mode = ModeLive
	attestedRequest.TargetID = mixtureTargetID("default")
	attestedRequest.SuiteIDs = []string{"live-mom-core"}
	attestedRequest.ChangeProfile = "recipe"
	first, err := service.CreateRunAs(context.Background(), ownerA, attestedRequest)
	if err != nil {
		t.Fatalf("create owner A: %v", err)
	}
	usageBeforeAttestation, err := service.LifecycleUsage(ownerA)
	if err != nil {
		t.Fatalf("usage before attestation: %v", err)
	}
	manifest, _, err := service.readDurableManifest(first.ID)
	if err != nil {
		t.Fatalf("read owner A manifest: %v", err)
	}
	if writeAttestationErr := service.store.writeExecutionAttestation(
		validExecutionAttestationForManifest(t, manifest),
	); writeAttestationErr != nil {
		t.Fatalf("write owned execution attestation fixture: %v", writeAttestationErr)
	}
	usageAfterAttestation, err := service.LifecycleUsage(ownerA)
	if err != nil {
		t.Fatalf("usage after attestation: %v", err)
	}
	attestationInfo, err := os.Lstat(filepath.Join(service.store.attestationRoot, first.ID+".json"))
	if err != nil {
		t.Fatalf("stat execution attestation: %v", err)
	}
	if usageAfterAttestation.Owners[0].ActualBytes-usageBeforeAttestation.Owners[0].ActualBytes != attestationInfo.Size() ||
		usageBeforeAttestation.Owners[0].ReservedBytes-usageAfterAttestation.Owners[0].ReservedBytes != attestationInfo.Size() {
		t.Fatalf(
			"execution attestation was not owner-charged: before=%+v after=%+v size=%d",
			usageBeforeAttestation.Owners[0], usageAfterAttestation.Owners[0], attestationInfo.Size(),
		)
	}
	if _, ownerQuotaErr := service.CreateRunAs(context.Background(), ownerA, validCreateRequest()); !errors.Is(ownerQuotaErr, ErrQuota) {
		t.Fatalf("owner run quota error=%v, want ErrQuota", ownerQuotaErr)
	}
	second, err := service.CreateRunAs(context.Background(), ownerB, validCreateRequest())
	if err != nil {
		t.Fatalf("create owner B: %v", err)
	}
	if _, storeQuotaErr := service.CreateRunAs(context.Background(), ownerC, validCreateRequest()); !errors.Is(storeQuotaErr, ErrQuota) {
		t.Fatalf("store quota error=%v, want ErrQuota", storeQuotaErr)
	}
	usageA, err := service.LifecycleUsage(ownerA)
	if err != nil {
		t.Fatalf("owner usage: %v", err)
	}
	usageAgain, err := service.LifecycleUsage(ownerA)
	if err != nil || !reflect.DeepEqual(usageA, usageAgain) || len(usageA.Owners) != 1 ||
		usageA.Owners[0].PrincipalDigest != ownerA.PrincipalDigest() {
		t.Fatalf("usage is not deterministic and owner-scoped: first=%+v second=%+v err=%v", usageA, usageAgain, err)
	}
	if evidenceQuotaErr := service.store.requireEvidenceQuotaUnlocked(first.ID, 0, 6*1024*1024, 0); !errors.Is(evidenceQuotaErr, ErrQuota) {
		t.Fatalf("evidence growth quota error=%v, want ErrQuota", evidenceQuotaErr)
	}
	if deleteOwnerBErr := service.DeleteRunAs(ownerB, second.ID); deleteOwnerBErr != nil {
		t.Fatalf("delete owner B: %v", deleteOwnerBErr)
	}
	if deleteOwnerAErr := service.DeleteRunAs(ownerA, first.ID); deleteOwnerAErr != nil {
		t.Fatalf("delete owner A: %v", deleteOwnerAErr)
	}
	for iteration := 0; iteration < 20; iteration++ {
		run, createErr := service.CreateRunAs(context.Background(), ownerA, validCreateRequest())
		if createErr != nil {
			t.Fatalf("repeated create %d: %v", iteration, createErr)
		}
		if deleteErr := service.DeleteRunAs(ownerA, run.ID); deleteErr != nil {
			t.Fatalf("repeated delete %d: %v", iteration, deleteErr)
		}
	}
	finalUsage, err := service.LifecycleUsage(testLifecycleActor(t, "admin", true))
	if err != nil || finalUsage.RunCount != 0 || finalUsage.ChargeableBytes > limits.MaxStoreBytes ||
		finalUsage.AuditBytes > limits.MaxAuditBytes {
		t.Fatalf("repeated growth escaped bounds: usage=%+v err=%v", finalUsage, err)
	}
}

func TestLifecyclePublicationFaultsAndConcurrentMutationFailClosed(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "owner", false)
	run, manifest := preparePendingRun(t, service, validCreateRequest())
	service.store.lifecycleAuditWriter = failingLifecycleAuditWriter{}
	if _, err := service.store.CreateBundleAs(owner, run, manifest); err == nil {
		t.Fatal("create succeeded after lifecycle audit failure")
	}
	if _, err := os.Lstat(filepath.Join(service.store.runsRoot, run.ID)); !os.IsNotExist(err) {
		t.Fatalf("failed create published a run bundle: %v", err)
	}
	service.store.lifecycleAuditWriter = atomicLifecycleAuditWriter{}
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create after audit recovery: %v", err)
	}
	before, err := service.RunLifecycle(owner, run.ID)
	if err != nil {
		t.Fatalf("read lifecycle before fault: %v", err)
	}
	service.store.lifecyclePersistence = failingLifecyclePersistence{}
	hold := true
	if _, publicationErr := service.UpdateRunLifecycle(owner, run.ID, UpdateLifecycleRequest{EvidenceHold: &hold}); publicationErr == nil {
		t.Fatal("lifecycle mutation survived injected publication failure")
	}
	service.store.lifecyclePersistence = atomicLifecyclePolicyPersistence{}
	after, err := service.RunLifecycle(owner, run.ID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("failed lifecycle publication changed policy: before=%+v after=%+v err=%v", before, after, err)
	}

	var wait sync.WaitGroup
	for index := 0; index < 40; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			value := index%2 == 0
			class := RetentionStandard
			if index%3 == 0 {
				class = RetentionEphemeral
			}
			_, _ = service.UpdateRunLifecycle(owner, run.ID, UpdateLifecycleRequest{
				EvidenceHold: &value, RetentionClass: &class,
			})
		}(index)
	}
	wait.Wait()
	if _, err := service.RunLifecycle(owner, run.ID); err != nil {
		t.Fatalf("concurrent lifecycle mutation corrupted policy: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before lifecycle mutation restart: %v", err)
	}
	if _, err := newStandaloneStore(service.store.root); err != nil {
		t.Fatalf("restart after concurrent lifecycle mutation: %v", err)
	}
}

func TestLifecyclePolicyV2RejectsUnpublishedIntermediateState(t *testing.T) {
	service, root := newLifecycleTestService(t, LifecycleLimits{})
	path := filepath.Join(service.store.lifecycleRoot, lifecyclePolicyFileName)
	var policy lifecycleStorePolicy
	if err := readJSON(path, &policy); err != nil {
		t.Fatalf("read current lifecycle policy: %v", err)
	}
	policy.SchemaVersion = "evaluation-lifecycle-policy.v1"
	policy.PolicyRevision = "evaluation-lifecycle-policy.2026-08-31"
	policy.PolicyDigest = lifecycleDigest(policy)
	if err := writeJSONAtomic(path, policy); err != nil {
		t.Fatalf("write intermediate lifecycle policy: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before intermediate policy restart: %v", err)
	}
	if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "fresh store") {
		t.Fatalf("intermediate policy error=%v, want explicit fresh-store ErrInvalid", err)
	}
}

func TestLifecycleRestartQuarantinesRunPolicyAndRejectsAuditCorruption(t *testing.T) {
	t.Run("run lifecycle quarantine", func(t *testing.T) {
		service, root := newTestService(t, &controlledProcess{}, 1)
		run, quarantineErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
		if quarantineErr != nil {
			t.Fatalf("CreateRun: %v", quarantineErr)
		}
		path := filepath.Join(root, "runs", run.ID, lifecycleFileName)
		if tamperLifecycleErr := os.WriteFile(path, []byte("{\"schema_version\":\"tampered\"}\n"), 0o600); tamperLifecycleErr != nil {
			t.Fatalf("tamper lifecycle: %v", tamperLifecycleErr)
		}
		if err := service.Close(); err != nil {
			t.Fatalf("close before lifecycle quarantine restart: %v", err)
		}
		restarted, quarantineErr := newStandaloneStore(root)
		if quarantineErr != nil {
			t.Fatalf("restart quarantined lifecycle: %v", quarantineErr)
		}
		ledger, quarantineErr := restarted.listRunLedger(SystemActor(), RunListQuery{Limit: 10})
		if quarantineErr != nil || ledger.LedgerComplete || ledger.WarningCount != 1 || len(ledger.Runs) != 0 {
			t.Fatalf("lifecycle corruption was not quarantined: ledger=%+v err=%v", ledger, quarantineErr)
		}
		if _, err := restarted.Usage(SystemActor()); !errors.Is(err, ErrConflict) {
			t.Fatalf("usage accepted an incomplete lifecycle ledger: %v", err)
		}
	})

	t.Run("audit fails startup", func(t *testing.T) {
		service, root := newTestService(t, &controlledProcess{}, 1)
		if _, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest()); err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		entries, err := os.ReadDir(service.store.lifecycleAuditRoot)
		if err != nil || len(entries) == 0 {
			t.Fatalf("read lifecycle audit: entries=%v err=%v", entries, err)
		}
		if err := os.WriteFile(filepath.Join(service.store.lifecycleAuditRoot, entries[0].Name()), []byte("{}\n"), 0o600); err != nil {
			t.Fatalf("tamper audit: %v", err)
		}
		if err := service.Close(); err != nil {
			t.Fatalf("close before audit corruption restart: %v", err)
		}
		if _, err := newStandaloneStore(root); err == nil {
			t.Fatal("restart accepted a corrupt lifecycle audit chain")
		}
	})

	t.Run("policy fails startup", func(t *testing.T) {
		service, root := newTestService(t, &controlledProcess{}, 1)
		if _, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest()); err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		policyPath := filepath.Join(service.store.lifecycleRoot, lifecyclePolicyFileName)
		if err := os.WriteFile(policyPath, []byte("{}\n"), 0o600); err != nil {
			t.Fatalf("tamper lifecycle policy: %v", err)
		}
		if err := service.Close(); err != nil {
			t.Fatalf("close before policy corruption restart: %v", err)
		}
		if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
			t.Fatalf("corrupt lifecycle policy error=%v, want ErrInvalid", err)
		}
	})

	t.Run("pre-contract store is rejected", func(t *testing.T) {
		root := t.TempDir()
		if err := os.Chmod(root, 0o700); err != nil {
			t.Fatalf("protect root: %v", err)
		}
		runs := filepath.Join(root, "runs")
		if err := os.Mkdir(runs, 0o700); err != nil {
			t.Fatalf("create runs: %v", err)
		}
		if err := os.Mkdir(filepath.Join(runs, newTestClientRequestID()), 0o700); err != nil {
			t.Fatalf("create legacy run: %v", err)
		}
		if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
			t.Fatalf("pre-contract store error=%v, want ErrInvalid", err)
		}
	})
}

func newLifecycleTestService(t *testing.T, limits LifecycleLimits) (*Service, string) {
	t.Helper()
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create private evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
		LifecycleLimits: limits,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	return service, root
}

func testLifecycleActor(t *testing.T, principal string, administrator bool) Actor {
	t.Helper()
	actor, err := NewActor(principal, administrator)
	if err != nil {
		t.Fatalf("NewActor: %v", err)
	}
	return actor
}

func requestForExistingRun(run Run) CreateRunRequest {
	return CreateRunRequest{
		ClientRequestID: run.ID, Name: run.Name, Description: run.Description,
		SuiteIDs: run.SuiteIDs, TrackIDs: run.TrackIDs, Mode: run.Mode,
		TargetID: run.TargetID, ChangeProfile: run.ChangeProfile,
		SampleLimit: run.SampleLimit, Concurrency: run.Concurrency,
		CapacitySLO:          copyCapacitySLO(run.CapacitySLO),
		CapacityLoadProtocol: copyCapacityLoadProtocol(run.CapacityLoadProtocol),
		Seed:                 run.Seed, BaselineRunID: run.BaselineRunID,
	}
}

func lifecycleAuditRecords(store *Store) []lifecycleAuditRecord {
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	records := make([]lifecycleAuditRecord, 0, len(store.lifecycle.records))
	for _, record := range store.lifecycle.records {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].Sequence < records[j].Sequence })
	return records
}

func assertLifecycleAuditDecision(t *testing.T, records []lifecycleAuditRecord, action, decision string) {
	t.Helper()
	for _, record := range records {
		if record.Action == action && record.Decision == decision {
			return
		}
	}
	t.Fatalf("lifecycle audit omitted %s/%s: %+v", action, decision, records)
}
