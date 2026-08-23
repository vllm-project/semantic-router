package quotareconciliation

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID = "11111111-1111-4111-8111-111111111111"
	testFenceID     = "22222222-2222-4222-8222-222222222222"
	testPrincipalID = "33333333-3333-4333-8333-333333333333"
	testOperationID = "44444444-4444-4444-8444-444444444444"
)

var testReconciliationNow = time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)

func TestListCursorIsBoundToNamespaceStateAndAuthorizedScope(t *testing.T) {
	repository := &reconciliationTestRepository{page: RepositoryPage{Items: []Fence{{
		ID: testFenceID, NamespaceID: testNamespaceID, State: FenceOpen,
		CreatedAt: testReconciliationNow,
	}}, HasMore: true}}
	service := newReconciliationTestService(t, repository, &reconciliationTestRuntime{}, testReconciliationNow)
	scope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(testNamespaceID),
		TeamIDs:     []accesscontrol.TeamID{"55555555-5555-4555-8555-555555555555"},
	}
	page, err := service.List(context.Background(), ListRequest{
		NamespaceID: testNamespaceID, State: FenceOpen, Scope: scope, PageSize: 25,
	})
	if err != nil || page.NextCursor == "" || !page.HasMore || repository.query.Limit != 25 {
		t.Fatalf("first page = %#v, query = %#v, err = %v", page, repository.query, err)
	}
	otherScope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(testNamespaceID),
		TeamIDs:     []accesscontrol.TeamID{"66666666-6666-4666-8666-666666666666"},
	}
	_, err = service.List(context.Background(), ListRequest{
		NamespaceID: testNamespaceID, State: FenceOpen, Scope: otherScope,
		Cursor: page.NextCursor, PageSize: 25,
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("cursor reused under another scope: %v", err)
	}
}

func TestReconcileRequiresEvidenceAndBindsIdempotentCommand(t *testing.T) {
	repository := &reconciliationTestRepository{enqueue: EnqueueResult{Operation: Operation{ID: testOperationID}}}
	service := newReconciliationTestService(t, repository, &reconciliationTestRuntime{}, testReconciliationNow)
	request := ReconcileRequest{
		NamespaceID: testNamespaceID, FenceID: testFenceID, ExpectedRevision: 3,
		IdempotencyKey: "unknown-fence-reconcile-0001", Strategy: StrategyConservativeDebit,
		Reason: "Use the immutable admission bound.",
		Actor:  Actor{PrincipalID: testPrincipalID, ActorChain: []string{testPrincipalID}, RequestID: "request-1"},
	}
	if _, err := service.Reconcile(context.Background(), request); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("missing evidence references error = %v", err)
	}
	request.EvidenceReferences = []string{"usage-ledger:event-1"}
	result, err := service.Reconcile(context.Background(), request)
	if err != nil || result.Operation.ID != testOperationID || repository.prepared.FenceID != testFenceID ||
		repository.command.Endpoint != reconcileEndpoint || repository.command.PrincipalID != testPrincipalID {
		t.Fatalf("reconcile = %#v, prepared = %#v, command = %#v, err = %v",
			result, repository.prepared, repository.command, err)
	}
}

func TestProcessOneKeepsFenceUntilRuntimeAndLedgerAreDurable(t *testing.T) {
	sharedCalls := []string{}
	repository := &reconciliationTestRepository{calls: &sharedCalls, found: true, claim: Claim{
		Phase: PhaseRuntimePending, PlanDigest: strings.Repeat("a", 64), LeaseOwner: "worker-1",
		LeaseToken: "77777777-7777-4777-8777-777777777777",
		Plan: Plan{
			ReconciliationID: "88888888-8888-4888-8888-888888888888",
			NamespaceID:      testNamespaceID, Partition: "partition-one", FenceID: testFenceID,
			AdmissionID: "admission-1", OperationID: testOperationID, Strategy: StrategyActual,
			Corrections: []quotaruntime.CounterCorrection{
				{BindingID: "binding-enforce", RuleID: "rule-1", Enforcement: quota.EnforcementEnforce},
				{BindingID: "binding-shadow", RuleID: "rule-2", Enforcement: quota.EnforcementShadow},
			},
		},
	}}
	runtime := &reconciliationTestRuntime{calls: &sharedCalls}
	service := newReconciliationTestService(t, repository, runtime, testReconciliationNow)
	processed, err := service.ProcessOne(context.Background())
	if err != nil || !processed {
		t.Fatalf("process one = %v, %v", processed, err)
	}
	want := []string{"claim", "runtime.apply", "repository.mark_runtime", "repository.settle_ledger", "runtime.remove_fence", "repository.complete"}
	if strings.Join(sharedCalls, ",") != strings.Join(want, ",") {
		t.Fatalf("saga order = %v, want %v", sharedCalls, want)
	}
	if len(runtime.removal.BindingIDs) != 1 || runtime.removal.BindingIDs[0] != "binding-enforce" {
		t.Fatalf("fence removal bindings = %v", runtime.removal.BindingIDs)
	}
}

func newReconciliationTestService(t *testing.T, repository Repository, runtime Runtime, now time.Time) *Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: repository, Runtime: runtime, WaiveAuth: reconciliationTestWaiveAuthenticator{},
		CommandCodec: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("k", 32)),
		}},
		IdempotencyTTL: time.Hour, WorkerID: "worker-1", WorkerConcurrency: 1,
		PollInterval: 10 * time.Millisecond, ClaimLease: time.Minute,
		Now: func() time.Time { return now }, NewID: func() string { return testOperationID },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

type reconciliationTestWaiveAuthenticator struct{}

func (reconciliationTestWaiveAuthenticator) AuthorizeWaive(context.Context, string, managementauth.LiveSession, time.Time) error {
	return nil
}

type reconciliationTestRepository struct {
	page     RepositoryPage
	query    FenceQuery
	enqueue  EnqueueResult
	command  managementcommand.Command
	prepared ReconcileRequest
	claim    Claim
	found    bool
	calls    *[]string
}

func (repository *reconciliationTestRepository) record(value string) {
	if repository.calls != nil {
		*repository.calls = append(*repository.calls, value)
	}
}

func (*reconciliationTestRepository) ReadyQuotaReconciliation(context.Context, *managementcommand.Codec) error {
	return nil
}

func (*reconciliationTestRepository) Get(context.Context, string, string) (Fence, error) {
	return Fence{}, nil
}

func (*reconciliationTestRepository) GetOperation(context.Context, string, string) (Operation, error) {
	return Operation{}, nil
}

func (repository *reconciliationTestRepository) List(_ context.Context, query FenceQuery) (RepositoryPage, error) {
	repository.query = query
	return repository.page, nil
}

func (repository *reconciliationTestRepository) Prepare(_ context.Context, command managementcommand.Command,
	request ReconcileRequest, _ string, _ time.Time,
) (EnqueueResult, error) {
	repository.command, repository.prepared = command, request
	return repository.enqueue, nil
}

func (repository *reconciliationTestRepository) Claim(context.Context, string, time.Time, time.Duration) (Claim, bool, error) {
	repository.record("claim")
	return repository.claim, repository.found, nil
}

func (repository *reconciliationTestRepository) MarkRuntimeApplied(context.Context, Claim, string, time.Time) error {
	repository.record("repository.mark_runtime")
	return nil
}

func (repository *reconciliationTestRepository) SettleLedger(context.Context, Claim, time.Time) error {
	repository.record("repository.settle_ledger")
	return nil
}

func (repository *reconciliationTestRepository) Complete(context.Context, Claim, time.Time) (Operation, error) {
	repository.record("repository.complete")
	return Operation{}, nil
}

func (*reconciliationTestRepository) Release(context.Context, Claim, time.Time, error) error {
	return nil
}

type reconciliationTestRuntime struct {
	calls   *[]string
	removal quotaruntime.FenceRemovalRequest
}

func (runtime *reconciliationTestRuntime) record(value string) {
	if runtime.calls != nil {
		*runtime.calls = append(*runtime.calls, value)
	}
}

func (runtime *reconciliationTestRuntime) ApplyReconciliation(context.Context, quotaruntime.ReconciliationRequest) (quotaruntime.ReconciliationResult, error) {
	runtime.record("runtime.apply")
	return quotaruntime.ReconciliationResult{StreamID: "1-0"}, nil
}

func (runtime *reconciliationTestRuntime) RemoveReconciledFence(_ context.Context, request quotaruntime.FenceRemovalRequest) (quotaruntime.MutationResult, error) {
	runtime.record("runtime.remove_fence")
	runtime.removal = request
	return quotaruntime.MutationResult{}, nil
}
