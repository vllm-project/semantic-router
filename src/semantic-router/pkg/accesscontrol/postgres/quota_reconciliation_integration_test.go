package postgres

import (
	"context"
	"database/sql"
	"net/netip"
	"os"
	"strings"
	"testing"
	"time"

	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestUnknownUsageReconciliationPostgresLifecycle(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("PostgreSQL control-plane test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	db := isolatedPolicyDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatal(err)
	}

	const (
		namespaceID = "a1111111-1111-4111-8111-111111111111"
		principalID = "a2222222-2222-4222-8222-222222222222"
		userID      = "a3333333-3333-4333-8333-333333333333"
		policyID    = "a4444444-4444-4444-8444-444444444444"
		ruleID      = "a5555555-5555-4555-8555-555555555555"
		bindingID   = "a6666666-6666-4666-8666-666666666666"
		fenceID     = "a7777777-7777-4777-8777-777777777777"
		eventID     = "a8888888-8888-4888-8888-888888888888"
		operationID = "a9999999-9999-4999-8999-999999999999"
		admissionID = "admission-reconciliation-one"
		dispatchID  = "dispatch-reconciliation-one"
	)
	now := time.Date(2026, 8, 23, 10, 0, 0, 0, time.UTC)
	evidenceDigest := strings.Repeat("ab", 32)
	seedUnknownUsageReconciliation(t, ctx, db, now, evidenceDigest, namespaceID, principalID,
		userID, policyID, ruleID, bindingID, fenceID, eventID, admissionID, dispatchID)

	store, testUnknownUsageReconciliationPostgresLifecycleErr := New(db)
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil {
		t.Fatal(testUnknownUsageReconciliationPostgresLifecycleErr)
	}
	codec, testUnknownUsageReconciliationPostgresLifecycleErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil {
		t.Fatal(testUnknownUsageReconciliationPostgresLifecycleErr)
	}
	request := quotareconciliation.ReconcileRequest{
		NamespaceID: namespaceID, FenceID: fenceID, ExpectedRevision: 1,
		IdempotencyKey: "reconcile-postgres-0001", Strategy: quotareconciliation.StrategyActual,
		Actual: &quotareconciliation.ActualUsage{
			Dispatches: []quotareconciliation.ActualDispatchUsage{{
				DispatchID: dispatchID, EvidenceDigest: evidenceDigest,
				InputTokens: "3", CacheReadTokens: "1", CacheWriteTokens: "0", OutputTokens: "2",
				Cost: quotareconciliation.Cost{Currency: "USD", Numerator: "1000000000000000"},
			}},
			ServedInputTokens: "3", ServedOutputTokens: "2",
		},
		EvidenceReferences: []string{"provider-receipt:one"}, Reason: "Apply authoritative provider usage.",
		Actor: quotareconciliation.Actor{
			PrincipalID: principalID, ActorChain: []string{principalID},
			RequestID: "reconciliation-request-one", SourceIP: netip.MustParseAddr("192.0.2.20"),
		},
	}
	payload := []byte(`{"fenceId":"` + fenceID + `","strategy":"actual"}`)
	command, testUnknownUsageReconciliationPostgresLifecycleErr := codec.Bind(managementcommand.NamespaceCommandScope(namespaceID), principalID,
		"/management/v1/unknown-usage-fences/{fenceId}:reconcile", request.IdempotencyKey,
		payload, now, now.Add(time.Hour))
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil {
		t.Fatal(testUnknownUsageReconciliationPostgresLifecycleErr)
	}
	enqueued, testUnknownUsageReconciliationPostgresLifecycleErr := store.Prepare(ctx, command, request, operationID, now)
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil || enqueued.Operation.ID != operationID || enqueued.Operation.State != quotareconciliation.OperationPending {
		t.Fatalf("Prepare() = %#v, %v", enqueued, testUnknownUsageReconciliationPostgresLifecycleErr)
	}
	claim, found, testUnknownUsageReconciliationPostgresLifecycleErr := store.Claim(ctx, "worker-one", now.Add(time.Second), time.Minute)
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil || !found || claim.Plan.OperationID != operationID || claim.Plan.Corrections[0].Amount != "7" ||
		!claim.Plan.Corrections[0].Known {
		t.Fatalf("Claim() = %#v, %v, %v", claim, found, testUnknownUsageReconciliationPostgresLifecycleErr)
	}
	if err := store.MarkRuntimeApplied(ctx, claim, "1-0", now.Add(2*time.Second)); err != nil {
		t.Fatal(err)
	}
	claim.Phase, claim.RuntimeStreamID = quotareconciliation.PhaseRuntimeApplied, "1-0"
	if err := store.SettleLedger(ctx, claim, now.Add(3*time.Second)); err != nil {
		t.Fatal(err)
	}
	claim.Phase = quotareconciliation.PhaseLedgerApplied
	completed, testUnknownUsageReconciliationPostgresLifecycleErr := store.Complete(ctx, claim, now.Add(4*time.Second))
	if testUnknownUsageReconciliationPostgresLifecycleErr != nil || completed.State != quotareconciliation.OperationSucceeded || completed.Completed != 1 {
		t.Fatalf("Complete() = %#v, %v", completed, testUnknownUsageReconciliationPostgresLifecycleErr)
	}

	assertReconciliationDurableState(t, ctx, db, namespaceID, admissionID, fenceID)
}

func assertReconciliationDurableState(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	namespaceID, admissionID, fenceID string,
) {
	t.Helper()
	var settlementState, fenceState string
	var correctionEvents, auditEvents int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT state FROM usage_settlements WHERE namespace_id=$1 AND admission_id=$2),
	  (SELECT state FROM unknown_usage_fences WHERE namespace_id=$1 AND id=$3::uuid),
	  (SELECT count(*) FROM usage_events WHERE namespace_id=$1 AND reconciliation_id IS NOT NULL),
	  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1 AND resource_id=$3::text)`,
		namespaceID, admissionID, fenceID,
	).Scan(&settlementState, &fenceState, &correctionEvents, &auditEvents); err != nil {
		t.Fatal(err)
	}
	if settlementState != "settled" || fenceState != "resolved" || correctionEvents != 1 || auditEvents < 3 {
		t.Fatalf("durable state settlement=%s fence=%s corrections=%d audits=%d",
			settlementState, fenceState, correctionEvents, auditEvents)
	}
}

func seedUnknownUsageReconciliation(t *testing.T, ctx context.Context, db *sql.DB, now time.Time,
	evidenceDigest, namespaceID, principalID, userID, policyID, ruleID, bindingID,
	fenceID, eventID, admissionID, dispatchID string,
) {
	t.Helper()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status) VALUES ($1,'reconciliation','partition-reconciliation','USD','active')`, []any{namespaceID}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,status) VALUES ($1,'test','reconciler','Reconciler','active')`, []any{principalID}},
		{`INSERT INTO access_subjects(namespace_id,id,kind) VALUES ($1,$2,'user')`, []any{namespaceID, userID}},
		{`INSERT INTO access_users(id,namespace_id,email,display_name,status)
VALUES ($1,$2,'reconciler@example.com','Reconciler','active')`, []any{userID, namespaceID}},
		{`INSERT INTO rate_limit_policies(id,namespace_id,name,status) VALUES ($1,$2,'Token limit','active')`, []any{policyID, namespaceID}},
		{`INSERT INTO rate_limit_rules
  (id,policy_id,metric,algorithm,limit_value,window_seconds,accounting,enforcement,ordinal)
VALUES ($1,$2,'total_tokens','sliding_log',100,60,'response_actual','enforce',0)`, []any{ruleID, policyID}},
		{`INSERT INTO rate_limit_bindings
  (id,namespace_id,policy_id,subject_id,binding_mode,quota_partition_id,status)
VALUES ($1,$2,$3,$4,'allocation','partition-reconciliation','active')`, []any{bindingID, namespaceID, policyID, userID}},
		{`INSERT INTO usage_settlements
  (namespace_id,admission_id,state,event_partition_date) VALUES ($1,$2,'unknown',$3)`, []any{namespaceID, admissionID, now.Format("2006-01-02")}},
		{`INSERT INTO usage_events
  (namespace_id,admission_id,event_date,event_id,event_kind,protocol,path,user_id,status_code,
   input_tokens,output_tokens,total_tokens,usage_state,costs,request_metadata,occurred_at,ingested_at)
VALUES ($1,$2,$3,$4,'unknown','openai.chat.v1','/v1/chat/completions',$5,200,
  1,1,2,'unknown','[{"currency":"USD","knownNumerator":"0","knownDispatches":"0","incompleteDispatches":"1"}]',
  jsonb_build_object('completedAt',$6::timestamptz),$6,$6)`, []any{namespaceID, admissionID, now.Format("2006-01-02"), eventID, userID, now}},
		{
			`INSERT INTO usage_dispatches
  (namespace_id,event_date,event_id,admission_id,dispatch_id,dispatch_ordinal,attempt_count,
   dispatch_type,input_tokens,cache_read_tokens,cache_write_tokens,output_tokens,usage_state,
   currency,evidence_digest,started_at)
VALUES ($1,$2,$3,$4,$5,0,1,'primary',0,0,0,0,'unknown','USD',decode($6,'hex'),$7)`,
			[]any{namespaceID, now.Format("2006-01-02"), eventID, admissionID, dispatchID, evidenceDigest, now},
		},
		{
			`INSERT INTO unknown_usage_fences
  (id,namespace_id,admission_id,reason,evidence,state)
VALUES ($1,$2,$3,'provider_usage_unavailable',
  jsonb_build_object('eventId',$4::text,'evidenceState','unknown','dispatches',jsonb_build_array(
    jsonb_build_object('dispatchId',$5::text,'reason','provider_usage_unavailable','evidenceDigest',$6::text))),'open')`,
			[]any{fenceID, namespaceID, admissionID, eventID, dispatchID, evidenceDigest},
		},
		{`INSERT INTO unknown_usage_fence_bindings
  (fence_id,binding_id,rule_id,admission_limit,maximum_debit,metric,algorithm,enforcement,
   window_seconds,unknown_dispatch_count,counter_incomplete_count)
VALUES ($1,$2,$3,100,80,'total_tokens','sliding_log','enforce',60,1,1)`, []any{fenceID, bindingID, ruleID}},
	}
	for _, statement := range statements {
		if _, err := tx.ExecContext(ctx, statement.query, statement.args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
}
