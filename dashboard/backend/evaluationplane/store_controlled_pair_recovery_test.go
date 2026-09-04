package evaluationplane

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type pausingLifecycleAuditWriter struct {
	entered chan string
	release chan struct{}
}

func (writer *pausingLifecycleAuditWriter) WriteExclusive(path string, value any) error {
	encoded, marshalErr := json.MarshalIndent(value, "", "  ")
	if marshalErr != nil {
		return marshalErr
	}
	encoded = append(encoded, '\n')
	temporary, createErr := os.CreateTemp(filepath.Dir(path), lifecycleAuditTempPrefix+"*")
	if createErr != nil {
		return createErr
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return err
	}
	_, writeErr := temporary.Write(encoded)
	if writeErr == nil {
		writeErr = temporary.Sync()
	}
	closeErr := temporary.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	writer.entered <- temporaryPath
	<-writer.release
	if err := os.Link(temporaryPath, path); err != nil {
		return err
	}
	return syncEvaluationDirectory(filepath.Dir(path), "paused lifecycle audit test")
}

func (writer *pausingLifecycleAuditWriter) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func TestControlledPairPublicationCrashRecoveryHasNoHalfVisiblePair(t *testing.T) {
	for _, failure := range controlledPairPublicationFailureCases() {
		t.Run(failure.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			failure.install(service.store, pair)
			_, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			)
			if err == nil {
				t.Fatalf("persistence failure %s did not interrupt publication", failure.name)
			}
			_, baselineErr := service.store.GetRun(pair.BaselineRunID)
			_, candidateErr := service.store.GetRun(pair.CandidateRunID)
			if failure.name == "publication_committed" {
				if baselineErr != nil || candidateErr != nil {
					t.Fatalf("committed pair is not fully visible: baseline=%v candidate=%v", baselineErr, candidateErr)
				}
			} else if !errors.Is(baselineErr, ErrNotFound) || !errors.Is(candidateErr, ErrNotFound) {
				t.Fatalf("uncommitted pair leaked to readers: baseline=%v candidate=%v", baselineErr, candidateErr)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before publication crash recovery: %v", err)
			}
			reopened, reopenErr := newStandaloneStore(root)
			if reopenErr != nil {
				t.Fatalf("recover publication failure %s: %v", failure.name, reopenErr)
			}
			assertStrictPendingPair(t, reopened, pair)
		})
	}
}

func TestControlledPairCrashBeforePublicationIntentRemovesOrphanStages(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	baselineMembership := controlledPairMembership{
		SchemaVersion: SchemaVersion, PairID: pair.PairID,
		RunID: pair.BaselineRunID, Role: controlledPairRoleBaseline,
	}
	candidateMembership := controlledPairMembership{
		SchemaVersion: SchemaVersion, PairID: pair.PairID,
		RunID: pair.CandidateRunID, Role: controlledPairRoleCandidate,
	}
	service.store.lifecycle.mu.Lock()
	service.store.lifecycle.evidenceMu.Lock()
	service.store.runIndex.coordinator.Lock()
	service.store.mu.Lock()
	items, prepareErr := service.store.prepareInitialBundlePublicationUnlocked(SystemActor(), []initialBundleSpec{
		{
			run: pair.BaselineRun, manifest: baselineManifest,
			lifecycle: newRunLifecycle(pair.BaselineRun, SystemActor()),
			decorate:  func(path string) error { return writeControlledPairMembership(path, baselineMembership) },
		},
		{
			run: pair.CandidateRun, manifest: candidateManifest,
			lifecycle: newRunLifecycle(pair.CandidateRun, SystemActor()),
			decorate:  func(path string) error { return writeControlledPairMembership(path, candidateMembership) },
		},
	}, 0)
	service.store.mu.Unlock()
	service.store.runIndex.coordinator.Unlock()
	service.store.lifecycle.evidenceMu.Unlock()
	service.store.lifecycle.mu.Unlock()
	if prepareErr != nil || len(items) != 2 {
		t.Fatalf("stage controlled pair before intent: count=%d err=%v", len(items), prepareErr)
	}
	for _, item := range items {
		if _, err := os.Lstat(item.stagedDir); err != nil {
			t.Fatalf("expected orphan stage before restart: %v", err)
		}
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before orphan stage recovery: %v", err)
	}
	reopened, reopenErr := newStandaloneStore(root)
	if reopenErr != nil {
		t.Fatalf("recover pre-intent controlled pair crash: %v", reopenErr)
	}
	assertControlledPairAbsent(t, reopened, pair)
}

func TestControlledPairStartCrashRecoveryNeverExposesRunningPendingSplit(t *testing.T) {
	for _, failure := range controlledPairStartFailureCases() {
		t.Run(failure.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatalf("publish pending controlled pair: %v", err)
			}
			failure.install(service.store, pair)
			service.store.lifecycle.mu.Lock()
			_, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if startErr == nil {
				t.Fatalf("persistence failure %s did not interrupt start", failure.name)
			}
			baseline, baselineErr := service.store.GetRun(pair.BaselineRunID)
			candidate, candidateErr := service.store.GetRun(pair.CandidateRunID)
			if baselineErr != nil || candidateErr != nil || baseline.Status != candidate.Status {
				t.Fatalf("pair reader observed split state: baseline=%+v/%v candidate=%+v/%v", baseline, baselineErr, candidate, candidateErr)
			}
			baselineEvents, _ := service.store.EventsAfter(pair.BaselineRunID, 0)
			candidateEvents, _ := service.store.EventsAfter(pair.CandidateRunID, 0)
			if len(baselineEvents) != len(candidateEvents) {
				t.Fatalf("pair reader observed split start receipt: baseline=%d candidate=%d", len(baselineEvents), len(candidateEvents))
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before start crash recovery: %v", err)
			}
			reopened, reopenErr := newStandaloneStore(root)
			if reopenErr != nil {
				t.Fatalf("recover start failure %s: %v", failure.name, reopenErr)
			}
			recoveredBaseline, _ := reopened.GetRun(pair.BaselineRunID)
			recoveredCandidate, _ := reopened.GetRun(pair.CandidateRunID)
			if recoveredBaseline.Status != recoveredCandidate.Status {
				t.Fatalf("recovered pair has split states: %s/%s", recoveredBaseline.Status, recoveredCandidate.Status)
			}
			if failure.name == "start_committed" {
				if recoveredBaseline.Status != StatusRunning {
					t.Fatalf("committed start did not roll forward: %s", recoveredBaseline.Status)
				}
			} else if recoveredBaseline.Status != StatusPending {
				t.Fatalf("uncommitted start did not roll back: %s", recoveredBaseline.Status)
			}
		})
	}
}

func TestControlledPairSubprocessCrashRecovery(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair before subprocess crash: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before controlled pair subprocess crash: %v", err)
	}
	//nolint:gosec // G204: the command is Go's current test binary with a compile-time test selector.
	command := exec.Command(os.Args[0], "-test.run=^TestControlledPairSubprocessCrashHelper$")
	command.Env = append(os.Environ(),
		"VLLM_SR_PAIR_CRASH_HELPER=1",
		"VLLM_SR_PAIR_CRASH_ROOT="+root,
		"VLLM_SR_PAIR_CRASH_ID="+pair.PairID,
	)
	err := command.Run()
	var exitError *exec.ExitError
	if !errors.As(err, &exitError) || exitError.ExitCode() != 93 {
		t.Fatalf("subprocess crash exit=%v, want code 93", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("recover subprocess crash: %v", err)
	}
	assertStrictPendingPair(t, reopened, pair)
}

func TestControlledPairSubprocessCrashAfterManifestTempSyncIsRecovered(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, _, _ := pendingControlledPairAggregate(t, service, SystemActor())
	if err := service.Close(); err != nil {
		t.Fatalf("close before manifest temp subprocess crash: %v", err)
	}
	//nolint:gosec // G204: the command is Go's current test binary with a compile-time test selector.
	command := exec.Command(os.Args[0], "-test.run=^TestControlledPairSubprocessCrashHelper$")
	command.Env = append(os.Environ(),
		"VLLM_SR_PAIR_CRASH_HELPER=1",
		"VLLM_SR_PAIR_TEMP_CRASH=1",
		"VLLM_SR_PAIR_CRASH_ROOT="+root,
		"VLLM_SR_PAIR_CRASH_ID="+pair.PairID,
	)
	if err := command.Run(); err == nil {
		t.Fatal("manifest temp crash helper exited successfully")
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("recover synced manifest temp: %v", err)
	}
	if _, err := reopened.readControlledPair(pair.PairID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("manifest temp crash published aggregate: %v", err)
	}
}

func TestControlledPairSubprocessCrashTempsAreRecoveredForEveryCanonicalState(t *testing.T) {
	for _, state := range []string{
		controlledPairStatePending,
		controlledPairStateRunning,
		controlledPairStateDeleting,
		controlledPairStateDeleted,
	} {
		t.Run(state, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatal(err)
			}
			switch state {
			case controlledPairStateRunning:
				service.store.lifecycle.mu.Lock()
				_, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
				service.store.lifecycle.mu.Unlock()
				if err != nil {
					t.Fatal(err)
				}
			case controlledPairStateDeleting:
				failAfterControlledPairManifestState(service.store, controlledPairStateDeleting)
				if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err == nil {
					t.Fatal("delete intent fault did not interrupt")
				}
			case controlledPairStateDeleted:
				if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err != nil {
					t.Fatal(err)
				}
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before canonical state subprocess crash: %v", err)
			}
			//nolint:gosec // G204: the command is Go's current test binary with a compile-time test selector.
			command := exec.Command(os.Args[0], "-test.run=^TestControlledPairSubprocessCrashHelper$")
			command.Env = append(os.Environ(),
				"VLLM_SR_PAIR_CRASH_HELPER=1",
				"VLLM_SR_PAIR_TEMP_CRASH=1",
				"VLLM_SR_PAIR_CRASH_ROOT="+root,
				"VLLM_SR_PAIR_CRASH_ID="+pair.PairID,
			)
			if err := command.Run(); err == nil {
				t.Fatal("canonical manifest temp crash helper exited successfully")
			}
			reopened, err := newStandaloneStore(root)
			if err != nil {
				t.Fatalf("recover %s atomic temp: %v", state, err)
			}
			recovered, err := reopened.readControlledPair(pair.PairID)
			if err != nil {
				t.Fatal(err)
			}
			want := state
			if state == controlledPairStateDeleting {
				want = controlledPairStateDeleted
			}
			if recovered.State != want {
				t.Fatalf("recovered state=%s, want %s", recovered.State, want)
			}
			entries, err := os.ReadDir(filepath.Join(reopened.controlledPairRoot, pair.PairID))
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), ".tmp-evaluation-") {
					t.Fatalf("recovery retained controlled pair temp %s", entry.Name())
				}
			}
		})
	}
}

func TestLifecycleAuditSubprocessCrashTempsRecoverDeterministically(t *testing.T) {
	for _, mode := range []string{"temp_synced", "linked"} {
		t.Run(mode, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			before := service.store.lifecycle.sequence
			if err := service.Close(); err != nil {
				t.Fatalf("close before lifecycle audit subprocess crash: %v", err)
			}
			//nolint:gosec // G204: the command is Go's current test binary with a compile-time test selector.
			command := exec.Command(os.Args[0], "-test.run=^TestControlledPairSubprocessCrashHelper$")
			command.Env = append(os.Environ(),
				"VLLM_SR_PAIR_CRASH_HELPER=1",
				"VLLM_SR_PAIR_AUDIT_CRASH="+mode,
				"VLLM_SR_PAIR_CRASH_ROOT="+root,
			)
			if err := command.Run(); err == nil {
				t.Fatal("audit crash helper exited successfully")
			}
			reopened, err := newStandaloneStore(root)
			if err != nil {
				t.Fatalf("recover audit %s crash: %v", mode, err)
			}
			entries, err := os.ReadDir(reopened.lifecycleAuditRoot)
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), lifecycleAuditTempPrefix) {
					t.Fatalf("recovery retained owned audit temp %s", entry.Name())
				}
			}
			want := before
			if mode == "linked" {
				want++
			}
			if reopened.lifecycle.sequence != want {
				t.Fatalf("recovered audit sequence=%d, want %d", reopened.lifecycle.sequence, want)
			}
		})
	}
}

func TestControlledPairSubprocessCrashHelper(t *testing.T) {
	if os.Getenv("VLLM_SR_PAIR_CRASH_HELPER") != "1" {
		t.Skip("subprocess helper")
	}
	store, openErr := newStandaloneStore(os.Getenv("VLLM_SR_PAIR_CRASH_ROOT"))
	if openErr != nil {
		t.Fatalf("open subprocess crash store: %v", openErr)
	}
	if os.Getenv("VLLM_SR_PAIR_TEMP_CRASH") == "1" {
		dir, dirErr := store.controlledPairDir(os.Getenv("VLLM_SR_PAIR_CRASH_ID"))
		if dirErr != nil {
			os.Exit(94)
		}
		if _, err := store.pairPersistence.EnsurePrivateDirectory(dir); err != nil {
			os.Exit(95)
		}
		if err := store.pairPersistence.SyncDirectory(store.controlledPairRoot, "test pair directory"); err != nil {
			os.Exit(96)
		}
		temp, tempErr := os.CreateTemp(dir, ".tmp-evaluation-*")
		if tempErr != nil {
			os.Exit(97)
		}
		_ = temp.Chmod(0o600)
		_, _ = temp.Write([]byte("synced manifest bytes"))
		_ = temp.Sync()
		_ = temp.Close()
		os.Exit(98)
	}
	if mode := os.Getenv("VLLM_SR_PAIR_AUDIT_CRASH"); mode != "" {
		temporary, err := os.CreateTemp(store.lifecycleAuditRoot, lifecycleAuditTempPrefix+"*")
		if err != nil {
			os.Exit(81)
		}
		if err := temporary.Chmod(0o600); err != nil {
			os.Exit(82)
		}
		if mode == "linked" {
			now := time.Now().UTC().Truncate(time.Microsecond)
			record := lifecycleAuditRecord{
				SchemaVersion: lifecycleAuditSchemaVersion, Sequence: store.lifecycle.sequence + 1,
				Timestamp: now, Action: "gc", Decision: "allowed", ReasonCode: "system",
				ResourceKind: lifecycleResourceStore,
				ActorDigest:  SystemActor().principalDigest, PreviousDigest: store.lifecycle.headDigest,
			}
			record.Digest = lifecycleAuditDigest(record)
			encoded, err := json.MarshalIndent(record, "", "  ")
			if err != nil {
				os.Exit(83)
			}
			if _, err := temporary.Write(append(encoded, '\n')); err != nil {
				os.Exit(84)
			}
			if err := temporary.Sync(); err != nil {
				os.Exit(85)
			}
			if err := temporary.Close(); err != nil {
				os.Exit(86)
			}
			name := fmt.Sprintf("%020d-%s.json", record.Sequence, trimSHA256(record.Digest))
			if err := os.Link(temporary.Name(), filepath.Join(store.lifecycleAuditRoot, name)); err != nil {
				os.Exit(87)
			}
			os.Exit(88)
		}
		if _, err := temporary.Write([]byte("durable staged audit")); err != nil {
			os.Exit(89)
		}
		if err := temporary.Sync(); err != nil {
			os.Exit(90)
		}
		if err := temporary.Close(); err != nil {
			os.Exit(91)
		}
		os.Exit(92)
	}
	pair, pairErr := store.readControlledPair(os.Getenv("VLLM_SR_PAIR_CRASH_ID"))
	if pairErr != nil {
		os.Exit(99)
	}
	store.statusPersistence = &exitingControlledPairRunStatusPersistence{
		runStatusPersistence: store.statusPersistence,
		runID:                pair.BaselineRunID, status: StatusRunning, exitCode: 93,
	}
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	_, _ = store.startControlledPairAs(SystemActor(), os.Getenv("VLLM_SR_PAIR_CRASH_ID"))
	t.Fatal("subprocess fault did not terminate")
}

func TestControlledPairStartingEventReaderStopsAtVisibleBoundary(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair: %v", err)
	}
	failAfterControlledPairRunStatus(service.store, pair.BaselineRunID, StatusRunning)
	service.store.lifecycle.mu.Lock()
	_, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if startErr == nil {
		t.Fatal("controlled pair status persistence failure did not interrupt transition")
	}
	eventsPath := filepath.Join(service.store.runsRoot, pair.BaselineRunID, eventsFileName)
	file, err := os.OpenFile(eventsPath, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatalf("open hidden event suffix: %v", err)
	}
	if _, err = file.WriteString(strings.Repeat("x", maxWorkerEventLineBytes+1) + "\n"); err != nil {
		_ = file.Close()
		t.Fatalf("write hidden event suffix: %v", err)
	}
	if err = file.Close(); err != nil {
		t.Fatalf("close hidden event suffix: %v", err)
	}
	events, err := service.store.EventsAfter(pair.BaselineRunID, 0)
	if err != nil {
		t.Fatalf("read visible starting events: %v", err)
	}
	if len(events) != 1 || events[0].ID != "1" {
		t.Fatalf("visible starting events=%+v, want only initial snapshot", events)
	}
}

func TestControlledPairAggregateCommitsTerminalStateAfterBothMembers(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish pending controlled pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	start, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if startErr != nil {
		t.Fatalf("commit controlled pair start: %v", startErr)
	}
	baseline, candidate := start.Baseline, start.Candidate
	baseline, _ = service.buildTerminalRun(baseline, errors.New("baseline failed"))
	candidate, _ = service.buildTerminalRun(candidate, errors.New("candidate failed"))
	if err := service.store.updateRunFixture(baseline); err != nil {
		t.Fatalf("persist baseline terminal state: %v", err)
	}
	if err := service.store.refreshControlledPairTerminalState(baseline.ID); err != nil {
		t.Fatalf("refresh half-terminal pair: %v", err)
	}
	half, _ := service.store.readControlledPair(pair.PairID)
	if half.State != controlledPairStateRunning {
		t.Fatalf("half-terminal pair left running aggregate state: %s", half.State)
	}
	if err := service.store.updateRunFixture(candidate); err != nil {
		t.Fatalf("persist candidate terminal state: %v", err)
	}
	if err := service.store.refreshControlledPairTerminalState(candidate.ID); err != nil {
		t.Fatalf("commit terminal aggregate: %v", err)
	}
	terminal, terminalErr := service.store.readControlledPair(pair.PairID)
	if terminalErr != nil || terminal.State != controlledPairStateTerminal ||
		!terminalStatus(terminal.BaselineRun.Status) || !terminalStatus(terminal.CandidateRun.Status) {
		t.Fatalf("terminal controlled pair aggregate=%+v err=%v", terminal, terminalErr)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before terminal pair restart: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen terminal controlled pair aggregate: %v", err)
	}
}

func TestControlledPairRecoveryRejectsRuntimeIdentityMutation(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*Run)
	}{
		{name: "membership", mutate: func(run *Run) { run.ControlledPair = nil }},
		{name: "started_at", mutate: func(run *Run) {
			changed := run.StartedAt.Add(time.Microsecond)
			run.StartedAt = &changed
		}},
		{name: "returned_to_pending", mutate: func(run *Run) {
			run.Status, run.StartedAt = StatusPending, nil
			run.Progress = RunProgress{Total: len(run.TrackIDs), Message: "Run created"}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
				t.Fatal(err)
			}
			service.store.lifecycle.mu.Lock()
			start, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			mutated := start.Baseline
			test.mutate(&mutated)
			if err := writeJSONAtomic(filepath.Join(service.store.runsRoot, mutated.ID, runFileName), mutated); err != nil {
				t.Fatalf("write runtime mutation: %v", err)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before runtime mutation restart: %v", err)
			}
			if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
				t.Fatalf("runtime mutation restart error=%v, want ErrInvalid", err)
			}
		})
	}
}

func TestControlledPairRecoveryRejectsSealedSourceBundleTamper(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*testing.T, *Service, string)
	}{
		{name: "report", mutate: func(t *testing.T, service *Service, runID string) {
			path := filepath.Join(service.store.runsRoot, runID, reportFileName)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, append(data, ' '), 0o600); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "private_receipt", mutate: func(t *testing.T, service *Service, runID string) {
			path := filepath.Join(service.store.runsRoot, runID, privateChecksumArtifactName)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "sealed_evidence", mutate: func(t *testing.T, service *Service, runID string) {
			path := filepath.Join(service.store.runsRoot, runID, "records.jsonl")
			if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatal(err)
			}
			test.mutate(t, service, pair.BaselineSourceRunID)
			if err := service.Close(); err != nil {
				t.Fatalf("close before sealed source restart: %v", err)
			}
			if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
				t.Fatalf("sealed source tamper restart error=%v, want ErrInvalid", err)
			}
		})
	}
}

func TestControlledPairRecoveryRejectsTombstoneIdentityTamper(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*controlledPairTombstone)
	}{
		{name: "owner", mutate: func(value *controlledPairTombstone) {
			value.OwnerPrincipalDigest = digestString("forged-owner")
		}},
		{name: "source", mutate: func(value *controlledPairTombstone) {
			value.BaselineSourceRunID = newTestClientRequestID()
		}},
		{name: "member", mutate: func(value *controlledPairTombstone) {
			value.CandidateRunID = newTestClientRequestID()
		}},
		{name: "deleted_at", mutate: func(value *controlledPairTombstone) {
			changed := value.DeletedAt.Add(time.Microsecond)
			value.DeletedAt = &changed
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatal(err)
			}
			if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(service.store.controlledPairRoot, pair.PairID, controlledPairTombstoneFile)
			var tombstone controlledPairTombstone
			if err := readJSON(path, &tombstone); err != nil {
				t.Fatal(err)
			}
			test.mutate(&tombstone)
			if err := writeJSONAtomic(path, tombstone); err != nil {
				t.Fatal(err)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before tombstone restart: %v", err)
			}
			if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
				t.Fatalf("tombstone identity tamper restart error=%v, want ErrInvalid", err)
			}
		})
	}
}

func TestControlledPairRecoveryRejectsMembershipRoleSwap(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(service.store.runsRoot, pair.CandidateRunID, controlledPairMembershipFile)
	forged := controlledPairMembership{
		SchemaVersion: SchemaVersion, PairID: pair.PairID,
		RunID: pair.CandidateRunID, Role: controlledPairRoleBaseline,
	}
	if err := writeJSONAtomic(path, forged); err != nil {
		t.Fatal(err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before membership restart: %v", err)
	}
	if _, err := newStandaloneStore(root); !errors.Is(err, ErrConflict) {
		t.Fatalf("swapped membership role restart error=%v, want ErrConflict", err)
	}
}

func TestControlledPairRecoveryRejectsStartingAndCancellingMutation(t *testing.T) {
	for _, test := range []struct {
		name    string
		prepare func(*Service, controlledPairManifest) error
	}{
		{
			name: "starting",
			prepare: func(service *Service, pair controlledPairManifest) error {
				failAfterControlledPairRunStatus(service.store, pair.BaselineRunID, StatusRunning)
				_, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
				return err
			},
		},
		{
			name: "cancelling",
			prepare: func(service *Service, pair controlledPairManifest) error {
				if _, err := service.store.startControlledPairAs(SystemActor(), pair.PairID); err != nil {
					return err
				}
				failAfterControlledPairRunStatus(service.store, pair.BaselineRunID, StatusCancelled)
				_, returnErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
				return returnErr
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatal(err)
			}
			service.store.lifecycle.mu.Lock()
			err := test.prepare(service, pair)
			service.store.lifecycle.mu.Unlock()
			if err == nil {
				t.Fatal("fault did not leave a recoverable transition")
			}
			physical, err := service.store.getRunPhysical(pair.BaselineRunID)
			if err != nil {
				t.Fatal(err)
			}
			physical.Description = "forged immutable identity"
			if err := writeJSONAtomic(
				filepath.Join(service.store.runsRoot, pair.BaselineRunID, runFileName), physical,
			); err != nil {
				t.Fatal(err)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before transitional mutation restart: %v", err)
			}
			if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) {
				t.Fatalf("%s mutation restart error=%v, want ErrInvalid", test.name, err)
			}
		})
	}
}
