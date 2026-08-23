package agentpublication

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type publicationStoreStub struct {
	reserve       func(int, string, int64) (agentmanagement.PublicationCommitReservation, error)
	finalize      func(int, string, int64) (agentmanagement.PublicationCommitResult, error)
	fail          func(int, string) (agentmanagement.PublicationCommitResult, error)
	reserveCalls  int
	finalizeCalls int
	failCalls     int
}

func (*publicationStoreStub) Ready(context.Context) error { return nil }

func (store *publicationStoreStub) ReservePublicationCommit(
	ctx context.Context, _, _ string, digest string, expected int64, _ agentmanagement.MutationContext,
) (agentmanagement.PublicationCommitReservation, error) {
	store.reserveCalls++
	if err := ctx.Err(); err != nil {
		return agentmanagement.PublicationCommitReservation{}, err
	}
	if store.reserve == nil {
		return agentmanagement.PublicationCommitReservation{}, nil
	}
	return store.reserve(store.reserveCalls, digest, expected)
}

func (store *publicationStoreStub) FinalizePublicationCommit(
	_ context.Context, _, _ string, operationID string, revision int64, _ time.Time,
) (agentmanagement.PublicationCommitResult, error) {
	store.finalizeCalls++
	if store.finalize == nil {
		return agentmanagement.PublicationCommitResult{}, nil
	}
	return store.finalize(store.finalizeCalls, operationID, revision)
}

func (store *publicationStoreStub) FailPublicationCommit(
	_ context.Context, _, _ string, code string, _ time.Time,
) (agentmanagement.PublicationCommitResult, error) {
	store.failCalls++
	if store.fail == nil {
		return agentmanagement.PublicationCommitResult{}, nil
	}
	return store.fail(store.failCalls, code)
}

type publicationPublisherStub struct {
	receipts []routingmanagement.RevisionReceipt
	errors   []error
	commands []managementcommand.Command
	calls    int
}

func (publisher *publicationPublisherStub) PublishEntrypoint(
	_ context.Context, _ string, _ string, _ int64, mutation routingmanagement.MutationContext,
) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	publisher.calls++
	if mutation.Command == nil {
		return nil, routingmanagement.RevisionReceipt{}, errors.New("publication command was not supplied")
	}
	publisher.commands = append(publisher.commands, *mutation.Command)
	index := publisher.calls - 1
	var receipt routingmanagement.RevisionReceipt
	if index < len(publisher.receipts) {
		receipt = publisher.receipts[index]
	}
	var err error
	if index < len(publisher.errors) {
		err = publisher.errors[index]
	}
	return &routingsnapshot.Snapshot{}, receipt, err
}

func TestCommitReservesPublishesAndFinalizesExactPlan(t *testing.T) {
	operationID := uuid.NewString()
	plan := publicationPlanFixture()
	store := &publicationStoreStub{
		reserve: func(_ int, digest string, expected int64) (agentmanagement.PublicationCommitReservation, error) {
			if digest != plan.Digest || expected != plan.Revision {
				t.Fatalf("reservation digest=%q revision=%d", digest, expected)
			}
			return agentmanagement.PublicationCommitReservation{Plan: plan}, nil
		},
		finalize: func(_ int, gotOperation string, revision int64) (agentmanagement.PublicationCommitResult, error) {
			if gotOperation != operationID || revision != 42 {
				t.Fatalf("finalize operation=%q revision=%d", gotOperation, revision)
			}
			return agentmanagement.PublicationCommitResult{Plan: plan, DesiredRevision: revision}, nil
		},
	}
	publisher := &publicationPublisherStub{receipts: []routingmanagement.RevisionReceipt{{
		OperationID: operationID, DesiredRevision: 42,
	}}}
	committer := newPublicationCommitter(t, store, publisher)
	request := publicationCommitRequest(plan)

	result, err := committer.Commit(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.OperationID != operationID || result.Revision != 42 || result.Replayed ||
		store.reserveCalls != 1 || store.finalizeCalls != 1 || publisher.calls != 1 {
		t.Fatalf("result=%+v reserve=%d publish=%d finalize=%d", result, store.reserveCalls, publisher.calls, store.finalizeCalls)
	}
	command := publisher.commands[0]
	if command.Endpoint != "/management/v1/publication-plans/"+plan.ID+":commit" ||
		command.PrincipalID == "" || command.Scope.NamespaceID != request.NamespaceID {
		t.Fatalf("bound command = %+v", command)
	}
}

func TestCommitReplaysCompletedReservationWithoutPublishing(t *testing.T) {
	plan := publicationPlanFixture()
	operationID := uuid.NewString()
	store := &publicationStoreStub{reserve: func(_ int, _ string, _ int64) (agentmanagement.PublicationCommitReservation, error) {
		return agentmanagement.PublicationCommitReservation{
			Plan: plan, OperationID: operationID, DesiredRevision: 31, Replayed: true,
		}, nil
	}}
	publisher := &publicationPublisherStub{}
	result, err := newPublicationCommitter(t, store, publisher).Commit(
		context.Background(), publicationCommitRequest(plan),
	)
	if err != nil || result.OperationID != operationID || result.Revision != 31 || !result.Replayed ||
		publisher.calls != 0 || store.finalizeCalls != 0 {
		t.Fatalf("result=%+v error=%v publish=%d finalize=%d", result, err, publisher.calls, store.finalizeCalls)
	}
}

func TestCommitRecoversCrashAfterRoutingCommitByReplayingSameCommand(t *testing.T) {
	plan := publicationPlanFixture()
	operationID := uuid.NewString()
	store := &publicationStoreStub{
		reserve: func(call int, _ string, _ int64) (agentmanagement.PublicationCommitReservation, error) {
			return agentmanagement.PublicationCommitReservation{Plan: plan, Replayed: call > 1}, nil
		},
		finalize: func(call int, _ string, _ int64) (agentmanagement.PublicationCommitResult, error) {
			if call == 1 {
				return agentmanagement.PublicationCommitResult{}, errors.New("simulated crash before Agent finalize")
			}
			return agentmanagement.PublicationCommitResult{Plan: plan, DesiredRevision: 55, Replayed: true}, nil
		},
	}
	publisher := &publicationPublisherStub{receipts: []routingmanagement.RevisionReceipt{
		{OperationID: operationID, DesiredRevision: 55},
		{OperationID: operationID, DesiredRevision: 55, Replayed: true},
	}}
	committer := newPublicationCommitter(t, store, publisher)
	request := publicationCommitRequest(plan)
	if _, err := committer.Commit(context.Background(), request); err == nil {
		t.Fatal("first commit unexpectedly survived the finalize crash")
	}
	result, err := committer.Commit(context.Background(), request)
	if err != nil || result.OperationID != operationID || result.Revision != 55 || !result.Replayed {
		t.Fatalf("recovered result=%+v error=%v", result, err)
	}
	if publisher.calls != 2 || len(publisher.commands) != 2 ||
		publisher.commands[0].ActiveDigest() != publisher.commands[1].ActiveDigest() ||
		publisher.commands[0].AdvisoryLockKey() != publisher.commands[1].AdvisoryLockKey() {
		t.Fatalf("publication replay did not reuse the exact command: %+v", publisher.commands)
	}
}

func TestCommitRejectsStaleETagDigestAndCancellationBeforePublish(t *testing.T) {
	plan := publicationPlanFixture()
	t.Run("stale etag", func(t *testing.T) {
		store := &publicationStoreStub{}
		publisher := &publicationPublisherStub{}
		request := publicationCommitRequest(plan)
		request.ExpectedETag = `"agent:0"`
		_, err := newPublicationCommitter(t, store, publisher).Commit(context.Background(), request)
		if !errors.Is(err, agentmanagement.ErrConflict) || store.reserveCalls != 0 || publisher.calls != 0 {
			t.Fatalf("error=%v reserve=%d publish=%d", err, store.reserveCalls, publisher.calls)
		}
	})
	for _, test := range []struct {
		name  string
		error error
	}{
		{name: "stale digest", error: agentmanagement.ErrApproval},
		{name: "cancel won reservation fence", error: agentmanagement.ErrApproval},
	} {
		t.Run(test.name, func(t *testing.T) {
			store := &publicationStoreStub{reserve: func(_ int, _ string, _ int64) (agentmanagement.PublicationCommitReservation, error) {
				return agentmanagement.PublicationCommitReservation{}, test.error
			}}
			publisher := &publicationPublisherStub{}
			_, err := newPublicationCommitter(t, store, publisher).Commit(context.Background(), publicationCommitRequest(plan))
			if !errors.Is(err, test.error) || publisher.calls != 0 || store.finalizeCalls != 0 {
				t.Fatalf("error=%v publish=%d finalize=%d", err, publisher.calls, store.finalizeCalls)
			}
		})
	}
	t.Run("canceled request context", func(t *testing.T) {
		store := &publicationStoreStub{}
		publisher := &publicationPublisherStub{}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := newPublicationCommitter(t, store, publisher).Commit(
			ctx, publicationCommitRequest(plan),
		)
		if !errors.Is(err, context.Canceled) || store.reserveCalls != 1 || publisher.calls != 0 ||
			store.finalizeCalls != 0 {
			t.Fatalf("error=%v reserve=%d publish=%d finalize=%d", err, store.reserveCalls, publisher.calls, store.finalizeCalls)
		}
	})
}

func TestCommitFinalizesDeterministicRoutingRejectionAsFailure(t *testing.T) {
	plan := publicationPlanFixture()
	store := &publicationStoreStub{
		reserve: func(_ int, _ string, _ int64) (agentmanagement.PublicationCommitReservation, error) {
			return agentmanagement.PublicationCommitReservation{Plan: plan}, nil
		},
		fail: func(_ int, code string) (agentmanagement.PublicationCommitResult, error) {
			if code != "publication_rejected" {
				t.Fatalf("failure code = %q", code)
			}
			return agentmanagement.PublicationCommitResult{Plan: plan}, nil
		},
	}
	publisher := &publicationPublisherStub{errors: []error{routingmanagement.ErrConflict}}
	_, err := newPublicationCommitter(t, store, publisher).Commit(context.Background(), publicationCommitRequest(plan))
	if !errors.Is(err, agentmanagement.ErrConflict) || store.failCalls != 1 || store.finalizeCalls != 0 {
		t.Fatalf("error=%v fail=%d finalize=%d", err, store.failCalls, store.finalizeCalls)
	}
}

func newPublicationCommitter(
	t *testing.T, store *publicationStoreStub, publisher *publicationPublisherStub,
) *Committer {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys:          map[string][]byte{"command-v1": []byte("0123456789abcdef0123456789abcdef")},
	})
	if err != nil {
		t.Fatal(err)
	}
	committer, err := New(Options{
		Store: store, Publisher: publisher, Commands: commands,
		Now: func() time.Time { return time.Date(2026, 8, 23, 9, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return committer
}

func publicationPlanFixture() agentmanagement.PublicationPlan {
	return agentmanagement.PublicationPlan{
		ID: uuid.NewString(), EntrypointID: "entrypoint-main",
		EntrypointResourceRevision: 9, Digest: "sha256:" + "a1" + repeatHex("00", 31),
		Revision: 7,
	}
}

func publicationCommitRequest(plan agentmanagement.PublicationPlan) managementserver.AgentPublicationCommitRequest {
	return managementserver.AgentPublicationCommitRequest{
		NamespaceID: uuid.NewString(), PlanID: plan.ID, PlanDigest: plan.Digest,
		ExpectedETag: `"agent:7"`, IdempotencyKey: "publication-commit-key-0001",
		Mutation: agentmanagement.MutationContext{PrincipalID: uuid.NewString(), RequestID: "request-1"},
	}
}

func repeatHex(value string, count int) string {
	result := ""
	for index := 0; index < count; index++ {
		result += value
	}
	return result
}
