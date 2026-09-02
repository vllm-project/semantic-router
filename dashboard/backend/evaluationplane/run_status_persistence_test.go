package evaluationplane

import (
	"context"
	"errors"
	"testing"
	"time"
)

type faultingRunStatusCommitPersistence struct {
	delegate      runStatusPersistence
	writeFailures int
	syncFailures  int
}

func (p *faultingRunStatusCommitPersistence) Write(path string, run Run) error {
	if p.writeFailures == 0 || !terminalStatus(run.Status) {
		return p.delegate.Write(path, run)
	}
	p.writeFailures--
	if err := publishJSONWithoutParentSync(path, run); err != nil {
		return err
	}
	return errors.New("injected terminal status sync failure after visible rename")
}

func (p *faultingRunStatusCommitPersistence) SyncDirectory(path, description string) error {
	if p.syncFailures > 0 {
		p.syncFailures--
		return errors.New("injected terminal status retry sync failure")
	}
	return p.delegate.SyncDirectory(path, description)
}

func TestTerminalRunRetryClosesVisibleStatusSyncFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("create terminal durability run: %v", createErr)
	}
	startedAt := time.Now().UTC().Truncate(time.Microsecond)
	run.Status, run.StartedAt = StatusRunning, &startedAt
	run.Progress.Message = "Run started"
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("start terminal durability run: %v", err)
	}
	completedAt := startedAt.Add(time.Second)
	run.Status, run.CompletedAt = StatusCancelled, &completedAt
	run.Progress.Message = "Run cancelled"
	faults := &faultingRunStatusCommitPersistence{
		delegate:      atomicRunStatusPersistence{},
		writeFailures: 1,
		syncFailures:  1,
	}
	service.store.statusPersistence = faults
	if _, err := service.store.commitTerminalRun(run); err == nil {
		t.Fatal("terminal transition succeeded after an uncertain visible status write")
	}
	if _, err := service.store.commitTerminalRun(run); err == nil {
		t.Fatal("terminal idempotent retry bypassed persistent directory sync failure")
	}
	if _, err := service.store.commitTerminalRun(run); err != nil {
		t.Fatalf("terminal idempotent retry did not close status durability: %v", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close terminal durability service: %v", err)
	}
	restarted := reopenTestService(t, root)
	durable, readErr := restarted.GetRunAs(SystemActor(), run.ID)
	if readErr != nil || durable.Status != StatusCancelled || durable.CompletedAt == nil ||
		!durable.CompletedAt.Equal(completedAt) {
		t.Fatalf("terminal status changed after restart: run=%+v err=%v", durable, readErr)
	}
}

func TestIdempotentStartClosesTerminalStatusSyncFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("create idempotent start durability run: %v", createErr)
	}
	startedAt := time.Now().UTC().Truncate(time.Microsecond)
	completedAt := startedAt.Add(time.Second)
	run.Status, run.StartedAt, run.CompletedAt = StatusCancelled, &startedAt, &completedAt
	run.Progress.Message = "Run cancelled"
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("prepare terminal idempotent start run: %v", err)
	}
	faults := &faultingRunStatusCommitPersistence{
		delegate:     atomicRunStatusPersistence{},
		syncFailures: 1,
	}
	service.store.statusPersistence = faults
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err == nil {
		t.Fatal("idempotent start bypassed terminal status directory sync failure")
	}
	if durable, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil || durable.Status != StatusCancelled {
		t.Fatalf("idempotent start did not close terminal status durability: run=%+v err=%v", durable, err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close idempotent start durability service: %v", err)
	}
	restarted := reopenTestService(t, root)
	durable, readErr := restarted.GetRunAs(SystemActor(), run.ID)
	if readErr != nil || durable.Status != StatusCancelled {
		t.Fatalf("idempotent start status changed after restart: run=%+v err=%v", durable, readErr)
	}
}

func TestWorkerProgressCannotResurrectRunCancelledByPeer(t *testing.T) {
	owner, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := owner.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("create ordinary progress race run: %v", createErr)
	}
	run = stageRunningTestRun(t, owner, run)
	peer := newSubscriberPeerService(t, owner)

	// Preserve the exact stale observation that a worker event used to carry
	// across its write. The peer terminalizes the shared durable Run before the
	// owner attempts to commit that worker's progress.
	stale, readErr := owner.store.GetRun(run.ID)
	if readErr != nil || stale.Status != StatusRunning {
		t.Fatalf("read stale running snapshot: run=%+v err=%v", stale, readErr)
	}
	cancelled, cancelErr := peer.CancelRunAs(SystemActor(), run.ID)
	if cancelErr != nil || cancelled.Status != StatusCancelled {
		t.Fatalf("peer cancel ordinary run: run=%+v err=%v", cancelled, cancelErr)
	}
	progress := stale.Progress
	progress.Percent = 50
	progress.Message = "Evaluation progress updated"
	if err := owner.store.commitWorkerProgress(stale.ID, progress); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale worker progress error=%v, want ErrConflict", err)
	}

	durable, durableErr := owner.store.GetRun(run.ID)
	if durableErr != nil || durable.Status != StatusCancelled || durable.CompletedAt == nil {
		t.Fatalf("peer cancellation was not preserved: run=%+v err=%v", durable, durableErr)
	}
}
