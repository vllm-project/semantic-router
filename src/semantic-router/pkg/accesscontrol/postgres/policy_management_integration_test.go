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
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	policyTestNamespaceID = "11111111-1111-4111-8111-111111111111"
	policyTestPrincipalID = "22222222-2222-4222-8222-222222222222"
	policyTestUserID      = "33333333-3333-4333-8333-333333333333"
	policyTestSecondUser  = "44444444-4444-4444-8444-444444444444"
	policyTestAPIKeyID    = "55555555-5555-4555-8555-555555555555"
	policyTestModelID     = "model_test"
)

func TestPolicyManagementPostgresEndToEnd(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL policy Management test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	db := isolatedPolicyDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatal(err)
	}

	seedPolicyManagement(t, ctx, db, policyTestNamespaceID, policyTestPrincipalID,
		policyTestUserID, policyTestSecondUser, policyTestAPIKeyID, policyTestModelID)

	store, testPolicyManagementPostgresEndToEndErr := New(db)
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	repository, testPolicyManagementPostgresEndToEndErr := NewPolicyManagementRepository(store)
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	commandCodec, testPolicyManagementPostgresEndToEndErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	service, testPolicyManagementPostgresEndToEndErr := policymanagement.NewService(policymanagement.Options{
		Repository: repository, CommandCodec: commandCodec,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "v1", Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("p", 32)),
		}}, IdempotencyTTL: time.Hour,
	})
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	t.Cleanup(service.Close)
	if err := service.Ready(ctx); err != nil {
		t.Fatal(err)
	}
	actor := policymanagement.Actor{
		PrincipalID: policyTestPrincipalID, ActorChain: []string{policyTestPrincipalID},
		RequestID: "policy-integration", SourceIP: netip.MustParseAddr("192.0.2.10"),
	}

	accessResult, secondAccess, exactAccessScope := assertAccessPolicyLifecycle(t, ctx, service, actor)
	allResults := accesscontrol.ResultScope{NamespaceID: policyTestNamespaceID, All: true}

	assertAccessPolicyBinding(t, ctx, service, actor, allResults, exactAccessScope, accessResult.ID)

	firstRate := createRatePolicy(t, ctx, service, policyTestNamespaceID, "Developer budget",
		"rate-policy-create-0001", "12", actor)
	secondRate := createRatePolicy(t, ctx, service, policyTestNamespaceID, "Admin budget",
		"rate-policy-create-0002", "120", actor)
	assertRatePolicyLifecycle(t, ctx, service, actor, allResults, firstRate.ID, secondRate.ID)

	inline := assertInlineRatePolicyLifecycle(t, ctx, db, service, actor)

	assertUsageFenceProtection(t, ctx, db, service, policyTestNamespaceID, policyTestPrincipalID, inline, actor)
	assertConcurrentAllocation(t, ctx, service, policyTestNamespaceID, policyTestSecondUser, firstRate.ID, secondRate.ID, actor)
	assertAPIKeyInlineTransactionSeam(t, ctx, db, policyTestNamespaceID, policyTestPrincipalID, policyTestSecondUser, actor)
	assertPolicyAccounting(t, ctx, db, service, actor, secondAccess.ID)
}

func assertAccessPolicyLifecycle(
	t *testing.T,
	ctx context.Context,
	service *policymanagement.Service,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, policymanagement.MutationResult, accesscontrol.ResultScope) {
	t.Helper()
	request := policymanagement.CreateAccessPolicyRequest{
		NamespaceID: policyTestNamespaceID, Name: "Developers", Description: "Model access",
		Status: accesscontrol.PolicyStatusActive,
		Grants: []policymanagement.AccessGrant{{
			ResourceType: accesscontrol.GrantResourceModel, ResourceID: policyTestModelID,
			Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
		}},
		IdempotencyKey: "access-policy-create-0001", Actor: actor,
	}
	created, createErr := service.CreateAccessPolicy(ctx, request)
	if createErr != nil || created.Replayed || created.Revision != 1 {
		t.Fatalf("create AccessPolicy = %#v, %v", created, createErr)
	}
	replayed, replayErr := service.CreateAccessPolicy(ctx, request)
	if replayErr != nil || !replayed.Replayed || replayed.ID != created.ID {
		t.Fatalf("replay AccessPolicy = %#v, %v", replayed, replayErr)
	}
	request.Name, request.Description, request.Grants = "Different", "", nil
	if _, err := service.CreateAccessPolicy(ctx, request); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting AccessPolicy replay error = %v", err)
	}
	missing := policymanagement.CreateAccessPolicyRequest{
		NamespaceID: policyTestNamespaceID, Name: "Broken grant", Status: accesscontrol.PolicyStatusActive,
		Grants: []policymanagement.AccessGrant{{
			ResourceType: accesscontrol.GrantResourceModel, ResourceID: "missing_model",
			Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow,
		}},
		IdempotencyKey: "access-policy-create-0002", Actor: actor,
	}
	if _, err := service.CreateAccessPolicy(ctx, missing); !errors.Is(err, policymanagement.ErrNotFound) {
		t.Fatalf("missing grant resource error = %v", err)
	}
	second, err := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: policyTestNamespaceID, Name: "Observers", Status: accesscontrol.PolicyStatusActive,
		IdempotencyKey: "access-policy-create-0003", Actor: actor,
	})
	if err != nil {
		t.Fatal(err)
	}
	all := accesscontrol.ResultScope{NamespaceID: policyTestNamespaceID, All: true}
	firstPage, err := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: policyTestNamespaceID, Status: accesscontrol.PolicyStatusActive, PageSize: 1, Scope: all,
	})
	if err != nil || len(firstPage.Items) != 1 || !firstPage.HasMore || firstPage.NextCursor == "" {
		t.Fatalf("first policy page = %#v, %v", firstPage, err)
	}
	secondPage, err := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: policyTestNamespaceID, Status: accesscontrol.PolicyStatusActive,
		PageSize: 1, Cursor: firstPage.NextCursor, Scope: all,
	})
	if err != nil || len(secondPage.Items) != 1 || secondPage.Items[0].ID == firstPage.Items[0].ID {
		t.Fatalf("second policy page = %#v, %v", secondPage, err)
	}
	searched, err := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: policyTestNamespaceID, Status: accesscontrol.PolicyStatusActive,
		Search: "DEVEL", PageSize: 1, Scope: all,
	})
	if err != nil || len(searched.Items) != 1 || searched.Items[0].ID != created.ID || searched.HasMore {
		t.Fatalf("searched AccessPolicy page = %#v, %v", searched, err)
	}
	exact := accesscontrol.ResultScope{
		NamespaceID: policyTestNamespaceID,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceAccessPolicy: {accesscontrol.ResourceID(created.ID)},
		},
	}
	narrow, err := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: policyTestNamespaceID, Status: accesscontrol.PolicyStatusActive,
		PageSize: 10, Scope: exact,
	})
	if err != nil || len(narrow.Items) != 1 || narrow.Items[0].ID != created.ID || narrow.HasMore {
		t.Fatalf("exact AccessPolicy scope page = %#v, %v", narrow, err)
	}
	return created, second, exact
}

func assertAccessPolicyBinding(
	t *testing.T,
	ctx context.Context,
	service *policymanagement.Service,
	actor policymanagement.Actor,
	all, exact accesscontrol.ResultScope,
	policyID string,
) {
	t.Helper()
	binding, err := service.CreateAccessBinding(ctx, policymanagement.CreateAccessBindingRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: policyID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: policyTestUserID},
		IdempotencyKey: "access-binding-create-0001", Actor: actor,
	})
	if err != nil || binding.Revision != 1 {
		t.Fatalf("create AccessPolicy binding = %#v, %v", binding, err)
	}
	page, err := service.ListAccessBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: policyTestNamespaceID, PageSize: 1, Scope: all,
	})
	if err != nil || len(page.Items) != 1 || page.Items[0].ID != binding.ID {
		t.Fatalf("AccessPolicy binding page = %#v, %v", page, err)
	}
	narrow, err := service.ListAccessBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: policyTestNamespaceID, PageSize: 10, Scope: exact,
	})
	if err != nil || len(narrow.Items) != 1 || narrow.Items[0].ID != binding.ID || narrow.HasMore {
		t.Fatalf("associated-policy binding scope page = %#v, %v", narrow, err)
	}
	if _, err := service.DeleteAccessPolicy(ctx, policymanagement.DeletePolicyRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: policyID, ExpectedRevision: 1, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrResourceInUse) {
		t.Fatalf("delete bound AccessPolicy error = %v", err)
	}
}

func assertRatePolicyLifecycle(
	t *testing.T,
	ctx context.Context,
	service *policymanagement.Service,
	actor policymanagement.Actor,
	scope accesscontrol.ResultScope,
	firstPolicyID, secondPolicyID string,
) {
	t.Helper()
	binding, bindingErr := service.CreateRateBinding(ctx, policymanagement.CreateRateBindingRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: firstPolicyID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: policyTestUserID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "rate-binding-create-0001", Actor: actor,
	})
	if bindingErr != nil || binding.Revision != 1 {
		t.Fatalf("create RateLimit binding = %#v, %v", binding, bindingErr)
	}
	page, pageErr := service.ListRateBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: policyTestNamespaceID, PageSize: 1, Scope: scope,
	})
	if pageErr != nil || len(page.Items) != 1 || page.Items[0].ID != binding.ID {
		t.Fatalf("RateLimit binding page = %#v, %v", page, pageErr)
	}
	if _, err := service.CreateRateBinding(ctx, policymanagement.CreateRateBindingRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: secondPolicyID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: policyTestUserID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "rate-binding-create-0002", Actor: actor,
	}); !errors.Is(err, policymanagement.ErrAllocationConflict) {
		t.Fatalf("second allocation error = %v", err)
	}
	current, err := service.GetRateLimitPolicy(ctx, policyTestNamespaceID, firstPolicyID)
	if err != nil {
		t.Fatal(err)
	}
	retainedRuleID := current.Rules[0].ID
	limitOnly := append([]policymanagement.RateLimitRule(nil), current.Rules...)
	limitOnly[0].Limit = "20"
	updated, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: firstPolicyID, ExpectedRevision: 1,
		Rules: &limitOnly, Actor: actor,
	})
	if err != nil || updated.Revision != 2 {
		t.Fatalf("limit-only update = %#v, %v", updated, err)
	}
	stored, err := service.GetRateLimitPolicy(ctx, policyTestNamespaceID, firstPolicyID)
	if err != nil || stored.Rules[0].ID != retainedRuleID || stored.Rules[0].Limit != "20" {
		t.Fatalf("retained rate rule = %#v, %v", stored.Rules, err)
	}
	semanticChange := append([]policymanagement.RateLimitRule(nil), stored.Rules...)
	semanticChange[0].Window = policymanagement.ISODuration(2 * time.Minute)
	if _, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: firstPolicyID, ExpectedRevision: 2,
		Rules: &semanticChange, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrCounterSemantics) {
		t.Fatalf("same-ID semantic update error = %v", err)
	}
	staleName := "Stale"
	if _, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: firstPolicyID, ExpectedRevision: 1,
		Name: &staleName, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrRevisionConflict) {
		t.Fatalf("stale RateLimitPolicy CAS error = %v", err)
	}
}

func assertInlineRatePolicyLifecycle(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *policymanagement.Service,
	actor policymanagement.Actor,
) policymanagement.InlineRateBindingResult {
	t.Helper()
	request := policymanagement.CreateInlineRateBindingRequest{
		NamespaceID: policyTestNamespaceID, Name: "Key override", Description: "Private allocation",
		Rules: []policymanagement.RateLimitRule{{
			Metric: accesscontrol.RateMetricTotalTokens, Algorithm: accesscontrol.RateAlgorithmCalendarWindow,
			Limit: "500000", CalendarPeriod: accesscontrol.CalendarPeriodDay, Timezone: "UTC",
			Accounting: accesscontrol.RateAccountingResponseActual, Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: policyTestAPIKeyID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "inline-rate-binding-0001", Actor: actor,
	}
	inline, err := service.CreateInlineRateBinding(ctx, request)
	if err != nil || !inline.Created || inline.Replayed || inline.Policy.ID != inline.Binding.PolicyID ||
		inline.Binding.QuotaPartitionID != "policy-test-partition" {
		t.Fatalf("inline policy/binding = %#v, %v", inline, err)
	}
	replayed, err := service.CreateInlineRateBinding(ctx, request)
	if err != nil || !replayed.Replayed || !replayed.Created ||
		replayed.Policy.ID != inline.Policy.ID || replayed.Binding.ID != inline.Binding.ID {
		t.Fatalf("replay inline policy/binding = %#v, %v", replayed, err)
	}
	var outbox, revisions int
	if err := db.QueryRowContext(ctx, `SELECT count(*), count(DISTINCT desired_revision)
FROM policy_outbox WHERE aggregate_id IN ($1,$2)`, inline.Policy.ID, inline.Binding.ID).
		Scan(&outbox, &revisions); err != nil || outbox != 2 || revisions != 1 {
		t.Fatalf("inline outbox/revisions = %d/%d, %v", outbox, revisions, err)
	}
	var policiesBefore, policiesAfter int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM rate_limit_policies`).Scan(&policiesBefore); err != nil {
		t.Fatal(err)
	}
	request.Name, request.Description, request.IdempotencyKey = "Conflicting override", "", "inline-rate-binding-0002"
	request.Rules = []policymanagement.RateLimitRule{{
		Metric: accesscontrol.RateMetricRequests, Algorithm: accesscontrol.RateAlgorithmSlidingLog,
		Limit: "1", Window: policymanagement.ISODuration(time.Minute),
		Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
	}}
	if _, err := service.CreateInlineRateBinding(ctx, request); !errors.Is(err, policymanagement.ErrAllocationConflict) {
		t.Fatalf("conflicting inline allocation error = %v", err)
	}
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM rate_limit_policies`).Scan(&policiesAfter); err != nil || policiesAfter != policiesBefore {
		t.Fatalf("rolled-back inline policy count = %d -> %d, %v", policiesBefore, policiesAfter, err)
	}
	return inline
}

func assertPolicyAccounting(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *policymanagement.Service,
	actor policymanagement.Actor,
	policyID string,
) {
	t.Helper()
	if _, err := service.DeleteAccessPolicy(ctx, policymanagement.DeletePolicyRequest{
		NamespaceID: policyTestNamespaceID, PolicyID: policyID, ExpectedRevision: 1, Actor: actor,
	}); err != nil {
		t.Fatalf("delete unbound AccessPolicy: %v", err)
	}
	var audits, outbox, unauditedRevisions int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_audit_events),
  (SELECT count(*) FROM policy_outbox),
  (SELECT count(*) FROM policy_revisions r
   WHERE EXISTS (SELECT 1 FROM policy_outbox o
                 WHERE o.namespace_id=r.namespace_id AND o.desired_revision=r.revision)
     AND NOT EXISTS (SELECT 1 FROM access_audit_events a
                     WHERE a.namespace_id=r.namespace_id AND a.desired_revision=r.revision))`).
		Scan(&audits, &outbox, &unauditedRevisions); err != nil {
		t.Fatal(err)
	}
	if audits == 0 || outbox < audits || unauditedRevisions != 0 {
		t.Fatalf("audit/outbox/unaudited accounting = %d/%d/%d", audits, outbox, unauditedRevisions)
	}
}

func createRatePolicy(
	t *testing.T,
	ctx context.Context,
	service *policymanagement.Service,
	namespaceID, name, idempotencyKey, limit string,
	actor policymanagement.Actor,
) policymanagement.MutationResult {
	t.Helper()
	result, err := service.CreateRateLimitPolicy(ctx, policymanagement.CreateRateLimitPolicyRequest{
		NamespaceID: namespaceID, Name: name, Status: accesscontrol.PolicyStatusActive,
		Rules: []policymanagement.RateLimitRule{{
			Metric:    accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: accesscontrol.QuotaValue(limit),
			Window: policymanagement.ISODuration(time.Minute), Accounting: accesscontrol.RateAccountingRequest,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		IdempotencyKey: idempotencyKey, Actor: actor,
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func assertUsageFenceProtection(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	service *policymanagement.Service,
	namespaceID, principalID string,
	inline policymanagement.InlineRateBindingResult,
	actor policymanagement.Actor,
) {
	t.Helper()
	fenceID := uuid.NewString()
	tx, assertUsageFenceProtectionErr := db.BeginTx(ctx, nil)
	if assertUsageFenceProtectionErr != nil {
		t.Fatal(assertUsageFenceProtectionErr)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, `INSERT INTO usage_settlements
  (namespace_id,admission_id,state,event_partition_date)
VALUES ($1,'admission-fenced','unknown','2026-08-22')`, namespaceID); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_fences
  (id,namespace_id,admission_id,reason,evidence,state,reconciliation_actor_id)
VALUES ($1,$2,'admission-fenced','backend timeout','{}','open',$3)`,
		fenceID, namespaceID, principalID); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_fence_bindings
  (fence_id,binding_id,rule_id,admission_limit,maximum_debit,
   metric,algorithm,enforcement,calendar_period,timezone,
   unknown_dispatch_count,counter_incomplete_count)
VALUES ($1,$2,$3,500000,500000,'total_tokens','calendar_window','enforce','day','UTC',1,1)`,
		fenceID, inline.Binding.ID, inline.Policy.Rules[0].ID); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, err := service.UpdateRateBinding(ctx, policymanagement.UpdateBindingRequest{
		NamespaceID: namespaceID, BindingID: inline.Binding.ID, ExpectedRevision: 1,
		Status: accesscontrol.BindingStatusDisabled, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrUnknownUsageFence) {
		t.Fatalf("fenced binding update error = %v", err)
	}
	name := "Renamed private allocation"
	metadataOnly, assertUsageFenceProtectionErr := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: inline.Policy.ID, ExpectedRevision: 1,
		Name: &name, Actor: actor,
	})
	if assertUsageFenceProtectionErr != nil || metadataOnly.Revision != 2 {
		t.Fatalf("fenced metadata-only update = %#v, %v", metadataOnly, assertUsageFenceProtectionErr)
	}
	policy, assertUsageFenceProtectionErr := service.GetRateLimitPolicy(ctx, namespaceID, inline.Policy.ID)
	if assertUsageFenceProtectionErr != nil {
		t.Fatal(assertUsageFenceProtectionErr)
	}
	rules := append([]policymanagement.RateLimitRule(nil), policy.Rules...)
	rules[0].Limit = "600000"
	limitOnly, assertUsageFenceProtectionErr := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: inline.Policy.ID, ExpectedRevision: 2,
		Rules: &rules, Actor: actor,
	})
	if assertUsageFenceProtectionErr != nil || limitOnly.Revision != 3 {
		t.Fatalf("fenced limit-only update = %#v, %v", limitOnly, assertUsageFenceProtectionErr)
	}
	rules[0].Enforcement = accesscontrol.RateEnforcementShadow
	if _, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: inline.Policy.ID, ExpectedRevision: 3,
		Rules: &rules, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrUnknownUsageFence) {
		t.Fatalf("fenced semantic rule update error = %v", err)
	}
	if _, err := service.DeleteRateBinding(ctx, policymanagement.DeleteBindingRequest{
		NamespaceID: namespaceID, BindingID: inline.Binding.ID, ExpectedRevision: 1, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrUnknownUsageFence) {
		t.Fatalf("fenced binding delete error = %v", err)
	}
}

func assertConcurrentAllocation(
	t *testing.T,
	ctx context.Context,
	service *policymanagement.Service,
	namespaceID, subjectID, firstPolicyID, secondPolicyID string,
	actor policymanagement.Actor,
) {
	t.Helper()
	policyIDs := []string{firstPolicyID, secondPolicyID}
	errorsSeen := make([]error, len(policyIDs))
	results := make([]policymanagement.MutationResult, len(policyIDs))
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index := range policyIDs {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			<-start
			results[index], errorsSeen[index] = service.CreateRateBinding(ctx,
				policymanagement.CreateRateBindingRequest{
					NamespaceID: namespaceID, PolicyID: policyIDs[index],
					Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: subjectID},
					Mode:           accesscontrol.RateBindingAllocation,
					IdempotencyKey: "concurrent-allocation-000" + string(rune('1'+index)), Actor: actor,
				})
		}(index)
	}
	close(start)
	wait.Wait()
	succeeded, conflicted := 0, 0
	for index, err := range errorsSeen {
		if err == nil && results[index].Revision == 1 {
			succeeded++
		} else if errors.Is(err, policymanagement.ErrAllocationConflict) {
			conflicted++
		} else {
			t.Fatalf("concurrent allocation %d = %#v, %v", index, results[index], err)
		}
	}
	if succeeded != 1 || conflicted != 1 {
		t.Fatalf("concurrent allocation succeeded/conflicted = %d/%d", succeeded, conflicted)
	}
}

func assertAPIKeyInlineTransactionSeam(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	namespaceID, principalID, ownerUserID string,
	actor policymanagement.Actor,
) {
	t.Helper()
	keyID, policyID, bindingID := uuid.NewString(), uuid.NewString(), uuid.NewString()
	ruleID := uuid.NewString()
	now := time.Now().UTC().Truncate(time.Microsecond)
	policy, assertAPIKeyInlineTransactionSeamErr := policymanagement.CompileInlineRateLimitPolicy(policymanagement.InlineRateLimitPolicySpec{
		NamespaceID: namespaceID, PolicyID: policyID, Name: "Issued key override",
		Rules: []policymanagement.RateLimitRule{{
			Metric:    accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "30",
			Window:      policymanagement.ISODuration(time.Minute),
			Accounting:  accesscontrol.RateAccountingRequest,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Now: now, NewRuleID: func() string { return ruleID },
	})
	if assertAPIKeyInlineTransactionSeamErr != nil {
		t.Fatal(assertAPIKeyInlineTransactionSeamErr)
	}
	tx, assertAPIKeyInlineTransactionSeamErr := db.BeginTx(ctx, nil)
	if assertAPIKeyInlineTransactionSeamErr != nil {
		t.Fatal(assertAPIKeyInlineTransactionSeamErr)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_subjects(namespace_id,id,kind,created_at)
VALUES ($1,$2,'api_key',$3)`, namespaceID, keyID, now); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_api_keys
  (id,namespace_id,name,owner_user_id,status,revision,created_at,updated_at)
VALUES ($1,$2,'transactional-key',$3,'active',1,$4,$4)`, keyID, namespaceID, ownerUserID, now); err != nil {
		t.Fatal(err)
	}
	materialized, assertAPIKeyInlineTransactionSeamErr := materializeManagementAPIKeyRateLimitOverride(ctx, tx, managedRateLimitOverride{
		InlinePolicy: &policy,
		Binding: policymanagement.RateLimitBinding{
			ID: bindingID, NamespaceID: namespaceID,
			Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: keyID},
			Mode:    accesscontrol.RateBindingAllocation, Status: accesscontrol.BindingStatusActive,
			Revision: 1, CreatedAt: now, UpdatedAt: now,
		},
	})
	if assertAPIKeyInlineTransactionSeamErr != nil || !materialized.Created || materialized.Policy.ID != policyID ||
		materialized.Binding.PolicyID != policyID {
		t.Fatalf("API-key inline seam = %#v, %v", materialized, assertAPIKeyInlineTransactionSeamErr)
	}
	keyMeta := MutationMeta{
		ActorPrincipalID: ptrPrincipal(principalID),
		ActorChain:       []accesscontrol.ManagementPrincipalID{accesscontrol.ManagementPrincipalID(principalID)},
		RequestID:        actor.RequestID, SourceIP: actor.SourceIP, Action: "api_key.create",
		Reason: "Create API key with inline quota.", Details: AuditDetails{},
	}
	policyMeta, assertAPIKeyInlineTransactionSeamErr := managedPolicyMutationMeta(actor, "rate_limit_policy.create", "Create inline policy.", nil)
	if assertAPIKeyInlineTransactionSeamErr != nil {
		t.Fatal(assertAPIKeyInlineTransactionSeamErr)
	}
	bindingMeta, assertAPIKeyInlineTransactionSeamErr := managedPolicyMutationMeta(actor, "rate_limit_binding.create", "Bind inline policy.", nil)
	if assertAPIKeyInlineTransactionSeamErr != nil {
		t.Fatal(assertAPIKeyInlineTransactionSeamErr)
	}
	if _, err := appendCompoundMutationRecords(ctx, tx, namespaceID, []compoundMutation{
		{Mutation: outboxMutation{
			AggregateType: "api_key", AggregateID: keyID,
			AggregateRevision: 1, Operation: outboxCreated,
		}, Meta: keyMeta},
		{Mutation: outboxMutation{
			AggregateType: "rate_limit_policy", AggregateID: policyID,
			AggregateRevision: 1, Operation: outboxCreated,
		}, Meta: policyMeta},
		{Mutation: outboxMutation{
			AggregateType: "rate_limit_binding", AggregateID: bindingID,
			AggregateRevision: 1, Operation: outboxCreated,
			References: managedRateBindingReferences(materialized.Binding),
		}, Meta: bindingMeta},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	var keys, policies, bindings, revisions int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_api_keys WHERE id=$1::uuid),
  (SELECT count(*) FROM rate_limit_policies WHERE id=$2::uuid),
  (SELECT count(*) FROM rate_limit_bindings WHERE id=$3::uuid),
  (SELECT count(DISTINCT desired_revision) FROM policy_outbox
   WHERE aggregate_id IN ($1::uuid::text,$2::uuid::text,$3::uuid::text))`,
		keyID, policyID, bindingID).Scan(&keys, &policies, &bindings, &revisions); err != nil ||
		keys != 1 || policies != 1 || bindings != 1 || revisions != 1 {
		t.Fatalf("atomic API-key resources/revisions = %d/%d/%d/%d, %v",
			keys, policies, bindings, revisions, err)
	}
}

func ptrPrincipal(value string) *accesscontrol.ManagementPrincipalID {
	principal := accesscontrol.ManagementPrincipalID(value)
	return &principal
}

func seedPolicyManagement(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	namespaceID, principalID, userID, secondUser, apiKeyID, modelID string,
) {
	t.Helper()
	catalogRevision := "sha256:" + strings.Repeat("a", 64)
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
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'policy-test','policy-test-partition','USD','active')`, []any{namespaceID}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,'test','policy-actor','Policy Actor','active')`, []any{principalID}},
		{`INSERT INTO provider_catalog_revisions
  (revision,snapshot_bytes,snapshot_digest,integration_references,catalog,
   required_wire_formats,required_credential_adapters,required_discovery_adapters)
VALUES ($1,decode('01','hex'),decode(repeat('01',32),'hex'),'[]','{}','[]','[]','[]')`, []any{catalogRevision}},
		{`INSERT INTO access_subjects(namespace_id,id,kind) VALUES
  ($1,$2,'user'),($1,$3,'user'),($1,$4,'api_key')`, []any{namespaceID, userID, secondUser, apiKeyID}},
		{`INSERT INTO access_users(id,namespace_id,email,display_name,status) VALUES
  ($1,$3,'first@example.com','First User','active'),
  ($2,$3,'second@example.com','Second User','active')`, []any{userID, secondUser, namespaceID}},
		{`INSERT INTO access_api_keys
  (id,namespace_id,name,owner_user_id,status) VALUES ($1,$2,'existing-key',$3,'active')`, []any{apiKeyID, namespaceID, userID}},
		{`INSERT INTO routing_models
  (id,namespace_id,name,status,current_revision) VALUES ($1,$2,'Test Model','active',1)`, []any{modelID, namespaceID}},
		{
			`INSERT INTO routing_model_revisions
  (model_id,revision,provider_catalog_revision,name,capabilities,reasoning,loras,execution,pricing,content_digest,created_by)
VALUES ($1,1,$2,'Test Model','{}','{}','[]','{}','{}',decode(repeat('02',32),'hex'),$3)`,
			[]any{modelID, catalogRevision, principalID},
		},
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

func isolatedPolicyDatabase(t *testing.T, ctx context.Context, dsn string) *sql.DB {
	t.Helper()
	admin, isolatedPolicyDatabaseErr := sql.Open("postgres", dsn)
	if isolatedPolicyDatabaseErr != nil {
		t.Fatal(isolatedPolicyDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_policy_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, isolatedPolicyDatabaseErr := url.Parse(dsn)
	if isolatedPolicyDatabaseErr != nil {
		t.Fatal(isolatedPolicyDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	database, isolatedPolicyDatabaseErr := sql.Open("postgres", parsed.String())
	if isolatedPolicyDatabaseErr != nil {
		t.Fatal(isolatedPolicyDatabaseErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}
