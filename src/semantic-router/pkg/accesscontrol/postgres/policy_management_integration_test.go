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

	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		principalID = "22222222-2222-4222-8222-222222222222"
		userID      = "33333333-3333-4333-8333-333333333333"
		secondUser  = "44444444-4444-4444-8444-444444444444"
		apiKeyID    = "55555555-5555-4555-8555-555555555555"
		modelID     = "model_test"
	)
	seedPolicyManagement(t, ctx, db, namespaceID, principalID, userID, secondUser, apiKeyID, modelID)

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
		PrincipalID: principalID, ActorChain: []string{principalID},
		RequestID: "policy-integration", SourceIP: netip.MustParseAddr("192.0.2.10"),
	}

	accessResult, testPolicyManagementPostgresEndToEndErr := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: "Developers", Description: "Model access",
		Status: accesscontrol.PolicyStatusActive,
		Grants: []policymanagement.AccessGrant{{
			ResourceType: accesscontrol.GrantResourceModel,
			ResourceID:   modelID, Permission: accesscontrol.GrantPermissionInvoke,
			Effect: accesscontrol.GrantEffectAllow,
		}},
		IdempotencyKey: "access-policy-create-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || accessResult.Replayed || accessResult.Revision != 1 {
		t.Fatalf("create AccessPolicy = %#v, %v", accessResult, testPolicyManagementPostgresEndToEndErr)
	}
	replayedAccess, testPolicyManagementPostgresEndToEndErr := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: "Developers", Description: "Model access",
		Status: accesscontrol.PolicyStatusActive,
		Grants: []policymanagement.AccessGrant{{
			ResourceType: accesscontrol.GrantResourceModel,
			ResourceID:   modelID, Permission: accesscontrol.GrantPermissionInvoke,
			Effect: accesscontrol.GrantEffectAllow,
		}},
		IdempotencyKey: "access-policy-create-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || !replayedAccess.Replayed || replayedAccess.ID != accessResult.ID {
		t.Fatalf("replay AccessPolicy = %#v, %v", replayedAccess, testPolicyManagementPostgresEndToEndErr)
	}
	if _, err := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: "Different", Status: accesscontrol.PolicyStatusActive,
		IdempotencyKey: "access-policy-create-0001", Actor: actor,
	}); !errors.Is(err, managementcommand.ErrConflict) {
		t.Fatalf("conflicting AccessPolicy replay error = %v", err)
	}
	if _, err := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: "Broken grant", Status: accesscontrol.PolicyStatusActive,
		Grants: []policymanagement.AccessGrant{{
			ResourceType: accesscontrol.GrantResourceModel,
			ResourceID:   "missing_model", Permission: accesscontrol.GrantPermissionDiscover,
			Effect: accesscontrol.GrantEffectAllow,
		}},
		IdempotencyKey: "access-policy-create-0002", Actor: actor,
	}); !errors.Is(err, policymanagement.ErrNotFound) {
		t.Fatalf("missing grant resource error = %v", err)
	}
	secondAccess, testPolicyManagementPostgresEndToEndErr := service.CreateAccessPolicy(ctx, policymanagement.CreateAccessPolicyRequest{
		NamespaceID: namespaceID, Name: "Observers", Status: accesscontrol.PolicyStatusActive,
		IdempotencyKey: "access-policy-create-0003", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	allResults := accesscontrol.ResultScope{NamespaceID: namespaceID, All: true}
	firstPage, testPolicyManagementPostgresEndToEndErr := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatusActive, PageSize: 1, Scope: allResults,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(firstPage.Items) != 1 || !firstPage.HasMore || firstPage.NextCursor == "" {
		t.Fatalf("first policy page = %#v, %v", firstPage, testPolicyManagementPostgresEndToEndErr)
	}
	secondPage, testPolicyManagementPostgresEndToEndErr := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatusActive,
		PageSize: 1, Cursor: firstPage.NextCursor, Scope: allResults,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(secondPage.Items) != 1 || secondPage.Items[0].ID == firstPage.Items[0].ID {
		t.Fatalf("second policy page = %#v, %v", secondPage, testPolicyManagementPostgresEndToEndErr)
	}
	searchedAccess, testPolicyManagementPostgresEndToEndErr := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatusActive,
		Search: "DEVEL", PageSize: 1, Scope: allResults,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(searchedAccess.Items) != 1 || searchedAccess.Items[0].ID != accessResult.ID || searchedAccess.HasMore {
		t.Fatalf("searched AccessPolicy page = %#v, %v", searchedAccess, testPolicyManagementPostgresEndToEndErr)
	}
	exactAccessScope := accesscontrol.ResultScope{
		NamespaceID: namespaceID,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceAccessPolicy: {accesscontrol.ResourceID(accessResult.ID)},
		},
	}
	narrowPolicies, testPolicyManagementPostgresEndToEndErr := service.ListAccessPolicies(ctx, policymanagement.ListPoliciesRequest{
		NamespaceID: namespaceID, Status: accesscontrol.PolicyStatusActive, PageSize: 10, Scope: exactAccessScope,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(narrowPolicies.Items) != 1 || narrowPolicies.Items[0].ID != accessResult.ID || narrowPolicies.HasMore {
		t.Fatalf("exact AccessPolicy scope page = %#v, %v", narrowPolicies, testPolicyManagementPostgresEndToEndErr)
	}

	accessBinding, testPolicyManagementPostgresEndToEndErr := service.CreateAccessBinding(ctx, policymanagement.CreateAccessBindingRequest{
		NamespaceID: namespaceID, PolicyID: accessResult.ID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: userID},
		IdempotencyKey: "access-binding-create-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || accessBinding.Revision != 1 {
		t.Fatalf("create AccessPolicy binding = %#v, %v", accessBinding, testPolicyManagementPostgresEndToEndErr)
	}
	accessBindingPage, testPolicyManagementPostgresEndToEndErr := service.ListAccessBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: namespaceID, PageSize: 1, Scope: allResults,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(accessBindingPage.Items) != 1 || accessBindingPage.Items[0].ID != accessBinding.ID {
		t.Fatalf("AccessPolicy binding page = %#v, %v", accessBindingPage, testPolicyManagementPostgresEndToEndErr)
	}
	narrowBindings, testPolicyManagementPostgresEndToEndErr := service.ListAccessBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: namespaceID, PageSize: 10, Scope: exactAccessScope,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(narrowBindings.Items) != 1 || narrowBindings.Items[0].ID != accessBinding.ID || narrowBindings.HasMore {
		t.Fatalf("associated-policy binding scope page = %#v, %v", narrowBindings, testPolicyManagementPostgresEndToEndErr)
	}
	if _, err := service.DeleteAccessPolicy(ctx, policymanagement.DeletePolicyRequest{
		NamespaceID: namespaceID, PolicyID: accessResult.ID, ExpectedRevision: 1, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrResourceInUse) {
		t.Fatalf("delete bound AccessPolicy error = %v", err)
	}

	firstRate := createRatePolicy(t, ctx, service, namespaceID, "Developer budget",
		"rate-policy-create-0001", "12", actor)
	secondRate := createRatePolicy(t, ctx, service, namespaceID, "Admin budget",
		"rate-policy-create-0002", "120", actor)
	rateBinding, testPolicyManagementPostgresEndToEndErr := service.CreateRateBinding(ctx, policymanagement.CreateRateBindingRequest{
		NamespaceID: namespaceID, PolicyID: firstRate.ID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: userID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "rate-binding-create-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || rateBinding.Revision != 1 {
		t.Fatalf("create RateLimit binding = %#v, %v", rateBinding, testPolicyManagementPostgresEndToEndErr)
	}
	rateBindingPage, testPolicyManagementPostgresEndToEndErr := service.ListRateBindings(ctx, policymanagement.ListBindingsRequest{
		NamespaceID: namespaceID, PageSize: 1, Scope: allResults,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || len(rateBindingPage.Items) != 1 || rateBindingPage.Items[0].ID != rateBinding.ID {
		t.Fatalf("RateLimit binding page = %#v, %v", rateBindingPage, testPolicyManagementPostgresEndToEndErr)
	}
	if _, err := service.CreateRateBinding(ctx, policymanagement.CreateRateBindingRequest{
		NamespaceID: namespaceID, PolicyID: secondRate.ID,
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: userID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "rate-binding-create-0002", Actor: actor,
	}); !errors.Is(err, policymanagement.ErrAllocationConflict) {
		t.Fatalf("second allocation error = %v", err)
	}

	currentRate, testPolicyManagementPostgresEndToEndErr := service.GetRateLimitPolicy(ctx, namespaceID, firstRate.ID)
	if testPolicyManagementPostgresEndToEndErr != nil {
		t.Fatal(testPolicyManagementPostgresEndToEndErr)
	}
	retainedRuleID := currentRate.Rules[0].ID
	limitOnly := append([]policymanagement.RateLimitRule(nil), currentRate.Rules...)
	limitOnly[0].Limit = "20"
	updatedRate, testPolicyManagementPostgresEndToEndErr := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: firstRate.ID, ExpectedRevision: 1,
		Rules: &limitOnly, Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || updatedRate.Revision != 2 {
		t.Fatalf("limit-only update = %#v, %v", updatedRate, testPolicyManagementPostgresEndToEndErr)
	}
	storedRate, testPolicyManagementPostgresEndToEndErr := service.GetRateLimitPolicy(ctx, namespaceID, firstRate.ID)
	if testPolicyManagementPostgresEndToEndErr != nil || storedRate.Rules[0].ID != retainedRuleID || storedRate.Rules[0].Limit != "20" {
		t.Fatalf("retained rate rule = %#v, %v", storedRate.Rules, testPolicyManagementPostgresEndToEndErr)
	}
	semanticChange := append([]policymanagement.RateLimitRule(nil), storedRate.Rules...)
	semanticChange[0].Window = policymanagement.ISODuration(2 * time.Minute)
	if _, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: firstRate.ID, ExpectedRevision: 2,
		Rules: &semanticChange, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrCounterSemantics) {
		t.Fatalf("same-ID semantic update error = %v", err)
	}
	staleName := "Stale"
	if _, err := service.UpdateRateLimitPolicy(ctx, policymanagement.UpdateRateLimitPolicyRequest{
		NamespaceID: namespaceID, PolicyID: firstRate.ID, ExpectedRevision: 1,
		Name: &staleName, Actor: actor,
	}); !errors.Is(err, policymanagement.ErrRevisionConflict) {
		t.Fatalf("stale RateLimitPolicy CAS error = %v", err)
	}

	inline, testPolicyManagementPostgresEndToEndErr := service.CreateInlineRateBinding(ctx, policymanagement.CreateInlineRateBindingRequest{
		NamespaceID: namespaceID, Name: "Key override", Description: "Private allocation",
		Rules: []policymanagement.RateLimitRule{{
			Metric:    accesscontrol.RateMetricTotalTokens,
			Algorithm: accesscontrol.RateAlgorithmCalendarWindow, Limit: "500000",
			CalendarPeriod: accesscontrol.CalendarPeriodDay, Timezone: "UTC",
			Accounting:  accesscontrol.RateAccountingResponseActual,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: apiKeyID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "inline-rate-binding-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || !inline.Created || inline.Replayed || inline.Policy.ID != inline.Binding.PolicyID ||
		inline.Binding.QuotaPartitionID != "policy-test-partition" {
		t.Fatalf("inline policy/binding = %#v, %v", inline, testPolicyManagementPostgresEndToEndErr)
	}
	replayedInline, testPolicyManagementPostgresEndToEndErr := service.CreateInlineRateBinding(ctx, policymanagement.CreateInlineRateBindingRequest{
		NamespaceID: namespaceID, Name: "Key override", Description: "Private allocation",
		Rules: []policymanagement.RateLimitRule{{
			Metric:    accesscontrol.RateMetricTotalTokens,
			Algorithm: accesscontrol.RateAlgorithmCalendarWindow, Limit: "500000",
			CalendarPeriod: accesscontrol.CalendarPeriodDay, Timezone: "UTC",
			Accounting:  accesscontrol.RateAccountingResponseActual,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: apiKeyID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "inline-rate-binding-0001", Actor: actor,
	})
	if testPolicyManagementPostgresEndToEndErr != nil || !replayedInline.Replayed || !replayedInline.Created ||
		replayedInline.Policy.ID != inline.Policy.ID || replayedInline.Binding.ID != inline.Binding.ID {
		t.Fatalf("replay inline policy/binding = %#v, %v", replayedInline, testPolicyManagementPostgresEndToEndErr)
	}
	var inlineOutbox, inlineRevisions int
	if err := db.QueryRowContext(ctx, `SELECT count(*), count(DISTINCT desired_revision)
FROM policy_outbox WHERE aggregate_id IN ($1,$2)`, inline.Policy.ID, inline.Binding.ID).
		Scan(&inlineOutbox, &inlineRevisions); err != nil || inlineOutbox != 2 || inlineRevisions != 1 {
		t.Fatalf("inline outbox/revisions = %d/%d, %v", inlineOutbox, inlineRevisions, err)
	}
	var policiesBefore, policiesAfter int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM rate_limit_policies`).Scan(&policiesBefore); err != nil {
		t.Fatal(err)
	}
	if _, err := service.CreateInlineRateBinding(ctx, policymanagement.CreateInlineRateBindingRequest{
		NamespaceID: namespaceID, Name: "Conflicting override",
		Rules: []policymanagement.RateLimitRule{{
			Metric:    accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "1",
			Window:      policymanagement.ISODuration(time.Minute),
			Accounting:  accesscontrol.RateAccountingRequest,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
		Subject:        policymanagement.Subject{Type: accesscontrol.SubjectKindAPIKey, ID: apiKeyID},
		Mode:           accesscontrol.RateBindingAllocation,
		IdempotencyKey: "inline-rate-binding-0002", Actor: actor,
	}); !errors.Is(err, policymanagement.ErrAllocationConflict) {
		t.Fatalf("conflicting inline allocation error = %v", err)
	}
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM rate_limit_policies`).Scan(&policiesAfter); err != nil || policiesAfter != policiesBefore {
		t.Fatalf("rolled-back inline policy count = %d -> %d, %v", policiesBefore, policiesAfter, err)
	}

	assertUsageFenceProtection(t, ctx, db, service, namespaceID, principalID, inline, actor)
	assertConcurrentAllocation(t, ctx, service, namespaceID, secondUser, firstRate.ID, secondRate.ID, actor)
	assertAPIKeyInlineTransactionSeam(t, ctx, db, namespaceID, principalID, secondUser, actor)

	if _, err := service.DeleteAccessPolicy(ctx, policymanagement.DeletePolicyRequest{
		NamespaceID: namespaceID, PolicyID: secondAccess.ID, ExpectedRevision: 1, Actor: actor,
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
	materialized, assertAPIKeyInlineTransactionSeamErr := materializeManagedAPIKeyRateLimitOverride(ctx, tx, managedRateLimitOverride{
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
