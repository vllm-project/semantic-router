package postgres

import (
	"context"
	"database/sql"
	"errors"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID  = "11111111-1111-4111-8111-111111111111"
	testPrincipalID  = "22222222-2222-4222-8222-222222222222"
	testFirstUserID  = "33333333-3333-4333-8333-333333333333"
	testSecondUserID = "44444444-4444-4444-8444-444444444444"
)

func TestPolicyBulkPostgresReplicaEnqueueRace(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	item := policybulk.AccessBindingItem{
		ItemID: uuid.NewString(), PolicyID: environment.createAccessPolicy(t, ctx),
		Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
	}
	request := policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID, Items: []policybulk.AccessBindingItem{item},
		IdempotencyKey: "replica-enqueue-race-0001", Actor: testActor("replica-race"),
	}
	services := []*policybulk.Service{
		environment.newBulkService(t, "replica-one", allowAllAuthorizer{}),
		environment.newBulkService(t, "replica-two", allowAllAuthorizer{}),
	}
	type response struct {
		result policybulk.EnqueueResult
		err    error
	}
	start := make(chan struct{})
	results := make(chan response, 2)
	var workers sync.WaitGroup
	for _, service := range services {
		workers.Add(1)
		go func(service *policybulk.Service) {
			defer workers.Done()
			<-start
			result, err := service.EnqueueAccessBindings(ctx, request)
			results <- response{result: result, err: err}
		}(service)
	}
	close(start)
	workers.Wait()
	close(results)
	var operationID string
	var replayed int
	for response := range results {
		if response.err != nil {
			t.Fatal(response.err)
		}
		if operationID == "" {
			operationID = response.result.Operation.ID
		} else if response.result.Operation.ID != operationID {
			t.Fatalf("replicas returned different operations: %s != %s", response.result.Operation.ID, operationID)
		}
		if response.result.Replayed {
			replayed++
		}
	}
	if replayed != 1 {
		t.Fatalf("replayed responses = %d, want 1", replayed)
	}
	var operations, items int
	if err := environment.db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_operations WHERE kind=$1),
  (SELECT count(*) FROM policy_bulk_operation_items WHERE operation_id=$2)`,
		policybulk.AccessBindingOperationKind, operationID).Scan(&operations, &items); err != nil {
		t.Fatal(err)
	}
	if operations != 1 || items != 1 {
		t.Fatalf("durable operations/items = %d/%d, want 1/1", operations, items)
	}
}

func TestPolicyBulkPostgresListKeysetFiltersAndCancelCASReplay(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	now := time.Now().UTC().Truncate(time.Microsecond)
	service, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := policybulk.NewService(policybulk.Options{
		Repository: environment.bulkRepository, Policies: environment.policyService,
		Authorization: allowAllAuthorizer{}, CommandCodec: environment.commands,
		CursorKeyring:  testOperationCursorKeyring(),
		IdempotencyTTL: time.Hour, WorkerID: "operation-reader", WorkerConcurrency: 1,
		PollInterval: 10 * time.Millisecond, ClaimLease: time.Minute, MaximumAttempts: 3,
		Now: func() time.Time { return now },
	})
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil {
		t.Fatal(testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	t.Cleanup(service.Close)
	policyID := environment.createAccessPolicy(t, ctx)
	enqueueAccess := func(key, itemID string) policybulk.Operation {
		t.Helper()
		result, err := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
			NamespaceID: testNamespaceID,
			Items: []policybulk.AccessBindingItem{{
				ItemID: itemID, PolicyID: policyID,
				Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
			}},
			IdempotencyKey: key, Actor: testActor(key),
		})
		if err != nil {
			t.Fatal(err)
		}
		return result.Operation
	}
	oldest := enqueueAccess("operation-list-access-oldest", uuid.NewString())
	now = now.Add(time.Second)
	_, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr = service.EnqueueRateBindings(ctx, policybulk.EnqueueRateRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.RateBindingItem{{
			ItemID: uuid.NewString(),
			InlinePolicy: &policybulk.InlineRateLimitPolicy{
				Name: "List filter quota",
				Rules: []policymanagement.RateLimitRule{{
					Metric:    accesscontrol.RateMetricRequests,
					Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "2",
					Window: policymanagement.ISODuration(time.Minute), Accounting: accesscontrol.RateAccountingRequest,
					Enforcement: accesscontrol.RateEnforcementEnforce,
				}},
			},
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
			Mode:    accesscontrol.RateBindingAllocation,
		}},
		IdempotencyKey: "operation-list-rate-middle", Actor: testActor("operation-list-rate-middle"),
	})
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil {
		t.Fatal(testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	now = now.Add(time.Second)
	newest := enqueueAccess("operation-list-access-newest", uuid.NewString())

	first, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := service.List(ctx, policybulk.ListRequest{
		NamespaceID: testNamespaceID,
		Kind:        policybulk.AccessBindingOperationKind, PageSize: 1, Visibility: testOperationVisibility(),
	})
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil || len(first.Items) != 1 || first.Items[0].ID != newest.ID || !first.HasMore || first.NextCursor == "" {
		t.Fatalf("first operation page = %#v, %v", first, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	second, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := service.List(ctx, policybulk.ListRequest{
		NamespaceID: testNamespaceID,
		Kind:        policybulk.AccessBindingOperationKind, PageSize: 1, Cursor: first.NextCursor,
		Visibility: testOperationVisibility(),
	})
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil || len(second.Items) != 1 || second.Items[0].ID != oldest.ID || second.HasMore {
		t.Fatalf("second operation page = %#v, %v", second, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	if _, err := service.List(ctx, policybulk.ListRequest{
		NamespaceID: testNamespaceID,
		Kind:        policybulk.AccessBindingOperationKind, State: policybulk.OperationPending,
		PageSize: 1, Cursor: first.NextCursor, Visibility: testOperationVisibility(),
	}); !errors.Is(err, policybulk.ErrInvalidRequest) {
		t.Fatalf("filter-mismatched cursor error = %v", err)
	}

	stale := policybulk.CancelRequest{
		NamespaceID: testNamespaceID, OperationID: newest.ID,
		ExpectedVersion: newest.Version + 1, IdempotencyKey: "operation-cancel-stale", Actor: testActor("cancel-stale"),
	}
	if _, err := service.Cancel(ctx, stale); !errors.Is(err, policybulk.ErrRevisionConflict) {
		t.Fatalf("stale cancel error = %v", err)
	}
	cancelRequest := policybulk.CancelRequest{
		NamespaceID: testNamespaceID, OperationID: newest.ID,
		ExpectedVersion: newest.Version, IdempotencyKey: "operation-cancel-replay", Actor: testActor("cancel-replay"),
	}
	cancelled, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := service.Cancel(ctx, cancelRequest)
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil || cancelled.Replayed || cancelled.Operation.State != policybulk.OperationCancelled ||
		cancelled.Operation.Version <= newest.Version {
		t.Fatalf("cancel result = %#v, %v", cancelled, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	replayed, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := service.Cancel(ctx, cancelRequest)
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil || !replayed.Replayed || replayed.Operation.Version != cancelled.Operation.Version {
		t.Fatalf("cancel replay = %#v, %v", replayed, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
	cancelledPage, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr := service.List(ctx, policybulk.ListRequest{
		NamespaceID: testNamespaceID,
		State:       policybulk.OperationCancelled, PageSize: 10, Visibility: testOperationVisibility(),
	})
	if testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr != nil || len(cancelledPage.Items) != 1 || cancelledPage.Items[0].ID != newest.ID {
		t.Fatalf("cancelled operation filter = %#v, %v", cancelledPage, testPolicyBulkPostgresListKeysetFiltersAndCancelCASReplayErr)
	}
}

func testOperationVisibility() policybulk.OperationVisibility {
	all := accesscontrol.ResultScope{NamespaceID: testNamespaceID, All: true}
	return policybulk.OperationVisibility{
		PrincipalID: testPrincipalID, Operation: all, Access: all, Rate: all,
	}
}

func TestPolicyBulkPostgresCrashLeaseReclaim(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	service := environment.newBulkService(t, "enqueue-replica", allowAllAuthorizer{})
	result, testPolicyBulkPostgresCrashLeaseReclaimErr := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.AccessBindingItem{{
			ItemID: uuid.NewString(), PolicyID: environment.createAccessPolicy(t, ctx),
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
		}},
		IdempotencyKey: "crash-reclaim-enqueue-0001", Actor: testActor("crash-reclaim"),
	})
	if testPolicyBulkPostgresCrashLeaseReclaimErr != nil {
		t.Fatal(testPolicyBulkPostgresCrashLeaseReclaimErr)
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	lease := 2 * time.Second
	first, found, testPolicyBulkPostgresCrashLeaseReclaimErr := environment.bulkRepository.Claim(ctx, "worker-before-crash", now, lease, 3)
	if testPolicyBulkPostgresCrashLeaseReclaimErr != nil || !found || first.Attempt != 1 {
		t.Fatalf("first claim = %#v/%t, %v", first, found, testPolicyBulkPostgresCrashLeaseReclaimErr)
	}
	if _, earlyFound, err := environment.bulkRepository.Claim(ctx, "early-worker", now.Add(time.Second), lease, 3); err != nil || earlyFound {
		t.Fatalf("early reclaim found=%t, err=%v", earlyFound, err)
	}
	second, found, testPolicyBulkPostgresCrashLeaseReclaimErr := environment.bulkRepository.Claim(ctx, "worker-after-crash", now.Add(lease+time.Microsecond), lease, 3)
	if testPolicyBulkPostgresCrashLeaseReclaimErr != nil || !found || second.Attempt != 2 || second.OperationID != result.Operation.ID {
		t.Fatalf("reclaimed claim = %#v/%t, %v", second, found, testPolicyBulkPostgresCrashLeaseReclaimErr)
	}
	completed, testPolicyBulkPostgresCrashLeaseReclaimErr := environment.bulkRepository.Complete(ctx, second,
		policybulk.ItemResult{BindingID: uuid.NewString()}, now.Add(lease+time.Second))
	if testPolicyBulkPostgresCrashLeaseReclaimErr != nil || completed.State != policybulk.OperationSucceeded {
		t.Fatalf("completed reclaimed operation = %#v, %v", completed, testPolicyBulkPostgresCrashLeaseReclaimErr)
	}
}

func TestPolicyBulkPostgresRetriesTransientExecutionAuthorization(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	now := time.Now().UTC().Truncate(time.Microsecond)
	authorizer := &transientAuthorizer{}
	service, err := policybulk.NewService(policybulk.Options{
		Repository: environment.bulkRepository, Policies: environment.policyService,
		Authorization: authorizer, CommandCodec: environment.commands,
		CursorKeyring:  testOperationCursorKeyring(),
		IdempotencyTTL: time.Hour, WorkerID: "retry-worker", WorkerConcurrency: 1,
		PollInterval: 10 * time.Millisecond, ClaimLease: time.Minute, MaximumAttempts: 3,
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	result, err := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.AccessBindingItem{{
			ItemID: uuid.NewString(), PolicyID: environment.createAccessPolicy(t, ctx),
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
		}},
		IdempotencyKey: "transient-authorization-retry-0001", Actor: testActor("authorization-retry"),
	})
	if err != nil {
		t.Fatal(err)
	}
	processed, err := service.ProcessOne(ctx)
	if !processed || err == nil || authorizer.calls != 1 {
		t.Fatalf("first transient attempt = processed %t, calls %d, error %v", processed, authorizer.calls, err)
	}
	pending, err := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if err != nil || pending.State != policybulk.OperationPending || pending.Completed != 0 {
		t.Fatalf("operation after transient failure = %#v, %v", pending, err)
	}
	now = now.Add(time.Second)
	processed, err = service.ProcessOne(ctx)
	if err != nil || !processed || authorizer.calls != 2 {
		t.Fatalf("retry attempt = processed %t, calls %d, error %v", processed, authorizer.calls, err)
	}
	completed, err := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if err != nil || completed.State != policybulk.OperationSucceeded || completed.Completed != 1 {
		t.Fatalf("operation after retry = %#v, %v", completed, err)
	}
}

func TestPolicyBulkPostgresExpiresAbandonedOperation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	now := time.Now().UTC().Truncate(time.Microsecond)
	service, testPolicyBulkPostgresExpiresAbandonedOperationErr := policybulk.NewService(policybulk.Options{
		Repository: environment.bulkRepository, Policies: environment.policyService,
		Authorization: allowAllAuthorizer{}, CommandCodec: environment.commands,
		CursorKeyring:  testOperationCursorKeyring(),
		IdempotencyTTL: time.Minute, WorkerID: "expiry-worker", WorkerConcurrency: 1,
		PollInterval: 10 * time.Millisecond, ClaimLease: time.Minute, MaximumAttempts: 3,
		Now: func() time.Time { return now },
	})
	if testPolicyBulkPostgresExpiresAbandonedOperationErr != nil {
		t.Fatal(testPolicyBulkPostgresExpiresAbandonedOperationErr)
	}
	t.Cleanup(service.Close)
	result, testPolicyBulkPostgresExpiresAbandonedOperationErr := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.AccessBindingItem{{
			ItemID: uuid.NewString(), PolicyID: environment.createAccessPolicy(t, ctx),
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
		}},
		IdempotencyKey: "abandoned-operation-expiry-0001", Actor: testActor("abandoned-expiry"),
	})
	if testPolicyBulkPostgresExpiresAbandonedOperationErr != nil {
		t.Fatal(testPolicyBulkPostgresExpiresAbandonedOperationErr)
	}
	if _, found, err := environment.bulkRepository.Claim(ctx, "expiry-reaper",
		now.Add(time.Minute+time.Microsecond), time.Minute, 3); err != nil || found {
		t.Fatalf("claim after operation expiry found=%t, err=%v", found, err)
	}
	expired, testPolicyBulkPostgresExpiresAbandonedOperationErr := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if testPolicyBulkPostgresExpiresAbandonedOperationErr != nil || expired.State != policybulk.OperationFailed || expired.Completed != 1 ||
		expired.Failed != 1 || len(expired.ItemErrors) != 1 || expired.ItemErrors[0].Code != "operation_expired" {
		t.Fatalf("expired operation = %#v, %v", expired, testPolicyBulkPostgresExpiresAbandonedOperationErr)
	}
}

func TestPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPath(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	policyID := environment.createAccessPolicy(t, ctx)
	deniedItemID := uuid.NewString()
	service := environment.newBulkService(t, "partial-worker", denyItemAuthorizer{itemID: deniedItemID})
	result, testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.AccessBindingItem{
			{
				ItemID: uuid.NewString(), PolicyID: policyID,
				Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
			},
			{
				ItemID: deniedItemID, PolicyID: policyID,
				Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testSecondUserID},
			},
		},
		IdempotencyKey: "partial-failure-enqueue-0001", Actor: testActor("partial-failure"),
	})
	if testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr != nil {
		t.Fatal(testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr)
	}
	for index := 0; index < 2; index++ {
		processed, err := service.ProcessOne(ctx)
		if err != nil || !processed {
			t.Fatalf("process item %d = %t, %v", index, processed, err)
		}
	}
	operation, testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr != nil {
		t.Fatal(testPolicyBulkPostgresPartialFailureUsesOrdinaryDomainPathErr)
	}
	if operation.State != policybulk.OperationPartiallySucceeded || operation.Completed != 2 ||
		operation.Failed != 1 || len(operation.ItemErrors) != 1 ||
		operation.ItemErrors[0].ItemID != deniedItemID || operation.ItemErrors[0].Code != "authorization_revoked" {
		t.Fatalf("partial operation = %#v", operation)
	}
	var bindings, audits, outbox int
	if err := environment.db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_policy_bindings WHERE policy_id=$1),
  (SELECT count(*) FROM access_audit_events WHERE action='access_policy_binding.create'),
  (SELECT count(*) FROM policy_outbox WHERE aggregate_type='access_policy_binding')`, policyID).
		Scan(&bindings, &audits, &outbox); err != nil {
		t.Fatal(err)
	}
	if bindings != 1 || audits != 1 || outbox != 1 {
		t.Fatalf("ordinary domain binding/audit/outbox = %d/%d/%d, want 1/1/1", bindings, audits, outbox)
	}
}

func TestPolicyBulkPostgresCancellationStopsPendingWork(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	policyID := environment.createAccessPolicy(t, ctx)
	service := environment.newBulkService(t, "cancel-worker", allowAllAuthorizer{})
	result, testPolicyBulkPostgresCancellationStopsPendingWorkErr := service.EnqueueAccessBindings(ctx, policybulk.EnqueueAccessRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.AccessBindingItem{
			{
				ItemID: uuid.NewString(), PolicyID: policyID,
				Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
			},
			{
				ItemID: uuid.NewString(), PolicyID: policyID,
				Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testSecondUserID},
			},
		},
		IdempotencyKey: "cancel-enqueue-0001", Actor: testActor("cancel-operation"),
	})
	if testPolicyBulkPostgresCancellationStopsPendingWorkErr != nil {
		t.Fatal(testPolicyBulkPostgresCancellationStopsPendingWorkErr)
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	claim, found, testPolicyBulkPostgresCancellationStopsPendingWorkErr := environment.bulkRepository.Claim(ctx, "running-worker", now, time.Second, 3)
	if testPolicyBulkPostgresCancellationStopsPendingWorkErr != nil || !found {
		t.Fatalf("running claim = %#v/%t, %v", claim, found, testPolicyBulkPostgresCancellationStopsPendingWorkErr)
	}
	current, testPolicyBulkPostgresCancellationStopsPendingWorkErr := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if testPolicyBulkPostgresCancellationStopsPendingWorkErr != nil {
		t.Fatal(testPolicyBulkPostgresCancellationStopsPendingWorkErr)
	}
	cancelling, testPolicyBulkPostgresCancellationStopsPendingWorkErr := service.Cancel(ctx, policybulk.CancelRequest{
		NamespaceID: testNamespaceID, OperationID: result.Operation.ID,
		ExpectedVersion: current.Version, IdempotencyKey: "cancel-operation-0001",
		Actor: testActor("cancel-operation-request"),
	})
	if testPolicyBulkPostgresCancellationStopsPendingWorkErr != nil || cancelling.Operation.State != policybulk.OperationRunning || cancelling.Operation.Completed != 1 {
		t.Fatalf("cancelling operation = %#v, %v", cancelling, testPolicyBulkPostgresCancellationStopsPendingWorkErr)
	}
	if _, found, err := environment.bulkRepository.Claim(ctx, "later-worker", now.Add(2*time.Second), time.Minute, 3); err != nil || found {
		t.Fatalf("claim after cancellation found=%t, err=%v", found, err)
	}
	terminal, testPolicyBulkPostgresCancellationStopsPendingWorkErr := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if testPolicyBulkPostgresCancellationStopsPendingWorkErr != nil || terminal.State != policybulk.OperationCancelled || terminal.Completed != 2 {
		t.Fatalf("cancelled operation = %#v, %v", terminal, testPolicyBulkPostgresCancellationStopsPendingWorkErr)
	}
	if claim.LeaseToken == "" {
		t.Fatal("running claim did not carry an ownership token")
	}
}

func TestPolicyBulkPostgresInlineRatePolicyIsOrdinaryAndAtomic(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	environment := newTestEnvironment(t, ctx)
	service := environment.newBulkService(t, "inline-rate-worker", allowAllAuthorizer{})
	result, err := service.EnqueueRateBindings(ctx, policybulk.EnqueueRateRequest{
		NamespaceID: testNamespaceID,
		Items: []policybulk.RateBindingItem{{
			ItemID: uuid.NewString(), InlinePolicy: &policybulk.InlineRateLimitPolicy{
				Name: "Interactive quota", Rules: []policymanagement.RateLimitRule{{
					Metric: accesscontrol.RateMetricRequests, Algorithm: accesscontrol.RateAlgorithmSlidingLog,
					Limit: "12", Window: policymanagement.ISODuration(time.Minute),
					Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
				}},
			}, Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: testFirstUserID},
			Mode: accesscontrol.RateBindingAllocation,
		}},
		IdempotencyKey: "inline-rate-enqueue-0001", Actor: testActor("inline-rate"),
	})
	if err != nil {
		t.Fatal(err)
	}
	processed, err := service.ProcessOne(ctx)
	if err != nil || !processed {
		t.Fatalf("process inline rate item = %t, %v", processed, err)
	}
	operation, err := service.Get(ctx, testNamespaceID, result.Operation.ID)
	if err != nil || operation.State != policybulk.OperationSucceeded {
		t.Fatalf("inline rate operation = %#v, %v", operation, err)
	}
	var policies, bindings, rules, compoundRevisionCount int
	if err := environment.db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM rate_limit_policies WHERE name='Interactive quota'),
  (SELECT count(*) FROM rate_limit_bindings b JOIN rate_limit_policies p ON p.id=b.policy_id
    WHERE p.name='Interactive quota'),
  (SELECT count(*) FROM rate_limit_rules r JOIN rate_limit_policies p ON p.id=r.policy_id
    WHERE p.name='Interactive quota' AND r.limit_value=12 AND r.window_seconds=60),
  (SELECT count(DISTINCT desired_revision) FROM policy_outbox
    WHERE aggregate_id IN (
      SELECT id::text FROM rate_limit_policies WHERE name='Interactive quota'
      UNION ALL
      SELECT b.id::text FROM rate_limit_bindings b JOIN rate_limit_policies p ON p.id=b.policy_id
      WHERE p.name='Interactive quota'))`).Scan(&policies, &bindings, &rules, &compoundRevisionCount); err != nil {
		t.Fatal(err)
	}
	if policies != 1 || bindings != 1 || rules != 1 || compoundRevisionCount != 1 {
		t.Fatalf("inline policy/binding/rule/revision = %d/%d/%d/%d, want 1/1/1/1",
			policies, bindings, rules, compoundRevisionCount)
	}
}

type testEnvironment struct {
	db             *sql.DB
	commands       *managementcommand.Codec
	policyService  *policymanagement.Service
	bulkRepository *Repository
}

func newTestEnvironment(t *testing.T, ctx context.Context) *testEnvironment {
	t.Helper()
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL policy bulk test database is not configured")
	}
	db := isolatedDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatal(err)
	}
	seedEnvironment(t, ctx, db)
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := accesspostgres.New(db)
	if err != nil {
		t.Fatal(err)
	}
	policyRepository, err := accesspostgres.NewPolicyManagementRepository(store)
	if err != nil {
		t.Fatal(err)
	}
	policyService, err := policymanagement.NewService(policymanagement.Options{
		Repository: policyRepository, CommandCodec: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("p", 32)),
		}}, IdempotencyTTL: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(policyService.Close)
	bulkRepository, err := NewRepository(db)
	if err != nil {
		t.Fatal(err)
	}
	if err := policyService.Ready(ctx); err != nil {
		t.Fatal(err)
	}
	if err := bulkRepository.Ready(ctx, commands); err != nil {
		t.Fatal(err)
	}
	return &testEnvironment{
		db: db, commands: commands, policyService: policyService,
		bulkRepository: bulkRepository,
	}
}

func (environment *testEnvironment) newBulkService(t *testing.T, workerID string,
	authorizer policybulk.ExecutionAuthorizer,
) *policybulk.Service {
	t.Helper()
	service, err := policybulk.NewService(policybulk.Options{
		Repository: environment.bulkRepository, Policies: environment.policyService,
		Authorization: authorizer, CommandCodec: environment.commands,
		CursorKeyring:  testOperationCursorKeyring(),
		IdempotencyTTL: time.Hour, WorkerID: workerID, WorkerConcurrency: 1,
		PollInterval: 10 * time.Millisecond, ClaimLease: time.Minute, MaximumAttempts: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func testOperationCursorKeyring() securitykeyring.Symmetric {
	return securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
		"v1": []byte(strings.Repeat("o", 32)),
	}}
}

func (environment *testEnvironment) createAccessPolicy(t *testing.T, ctx context.Context) string {
	t.Helper()
	result, err := environment.policyService.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: testNamespaceID, Name: "Policy " + uuid.NewString(),
		Status: accesscontrol.PolicyStatusActive, IdempotencyKey: "create-policy-" + uuid.NewString(),
		Actor: testActor("create-policy"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return result.ID
}

type allowAllAuthorizer struct{}

func (allowAllAuthorizer) AuthorizePolicyBulkItem(context.Context, policybulk.AuthorizationRequest) error {
	return nil
}

type denyItemAuthorizer struct{ itemID string }

func (authorizer denyItemAuthorizer) AuthorizePolicyBulkItem(_ context.Context,
	request policybulk.AuthorizationRequest,
) error {
	if request.ItemID == authorizer.itemID {
		return policybulk.ErrExecutionDenied
	}
	return nil
}

type transientAuthorizer struct{ calls int }

func (authorizer *transientAuthorizer) AuthorizePolicyBulkItem(
	_ context.Context,
	_ policybulk.AuthorizationRequest,
) error {
	authorizer.calls++
	if authorizer.calls == 1 {
		return policymanagement.ErrUnavailable
	}
	return nil
}

func testActor(requestID string) policymanagement.Actor {
	return policymanagement.Actor{
		PrincipalID: testPrincipalID, ActorChain: []string{testPrincipalID},
		RequestID: requestID, SourceIP: netip.MustParseAddr("192.0.2.20"),
	}
}

func seedEnvironment(t *testing.T, ctx context.Context, db *sql.DB) {
	t.Helper()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'policy-bulk-test','policy-bulk-partition','USD','active')`, []any{testNamespaceID}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,'test','policy-bulk-actor','Policy Bulk Actor','active')`, []any{testPrincipalID}},
		{`INSERT INTO access_subjects(namespace_id,id,kind) VALUES
  ($1,$2,'user'),($1,$3,'user')`, []any{testNamespaceID, testFirstUserID, testSecondUserID}},
		{`INSERT INTO access_users(id,namespace_id,email,display_name,status) VALUES
  ($1,$3,'first@example.com','First User','active'),
  ($2,$3,'second@example.com','Second User','active')`, []any{testFirstUserID, testSecondUserID, testNamespaceID}},
	}
	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement.query, statement.args...); err != nil {
			t.Fatal(err)
		}
	}
}

func isolatedDatabase(t *testing.T, ctx context.Context, dsn string) *sql.DB {
	t.Helper()
	admin, isolatedDatabaseErr := sql.Open("postgres", dsn)
	if isolatedDatabaseErr != nil {
		t.Fatal(isolatedDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_policy_bulk_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, isolatedDatabaseErr := url.Parse(dsn)
	if isolatedDatabaseErr != nil {
		t.Fatal(isolatedDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	database, isolatedDatabaseErr := sql.Open("postgres", parsed.String())
	if isolatedDatabaseErr != nil {
		t.Fatal(isolatedDatabaseErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}
