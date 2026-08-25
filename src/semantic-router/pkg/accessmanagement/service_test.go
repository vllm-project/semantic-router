package accessmanagement

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type repositoryStub struct {
	snapshot PolicySnapshot
	exists   bool
	mutation RoutingContextMutation
	loadErr  error
	update   UpdateRoutingContextRequest
}

func (stub *repositoryStub) Ready(context.Context) error { return nil }

func (stub *repositoryStub) LoadPolicySnapshot(context.Context, string, Subject) (PolicySnapshot, error) {
	return stub.snapshot, stub.loadErr
}

func (stub *repositoryStub) ResourceExists(context.Context, string, accesscontrol.GrantResource) (bool, error) {
	return stub.exists, nil
}

func (stub *repositoryStub) UpdateRoutingContext(_ context.Context, request UpdateRoutingContextRequest) (RoutingContextMutation, error) {
	stub.update = request
	return stub.mutation, nil
}

type appliedStub struct {
	policy accessruntime.AppliedPolicy
	err    error
}

type routingSnapshotStub struct {
	snapshot *routingsnapshot.Snapshot
	err      error
}

func (stub *routingSnapshotStub) ReadRoutingSnapshot(context.Context, string, int64) (*routingsnapshot.Snapshot, error) {
	return stub.snapshot, stub.err
}

func (stub *appliedStub) ReadAppliedPolicy(context.Context, string, string, string) (accessruntime.AppliedPolicy, error) {
	return stub.policy, stub.err
}

type meterStub struct {
	result   quotaruntime.MeterReadResult
	err      error
	requests []quotaruntime.MeterReadRequest
}

func (stub *meterStub) ReadMeters(_ context.Context, request quotaruntime.MeterReadRequest) (quotaruntime.MeterReadResult, error) {
	stub.requests = append(stub.requests, request)
	return stub.result, stub.err
}

type waiterStub struct {
	namespace string
	partition string
	revision  uint64
	err       error
}

func (stub *waiterStub) WaitApplied(_ context.Context, namespace, partition string, revision uint64) error {
	stub.namespace, stub.partition, stub.revision = namespace, partition, revision
	return stub.err
}

func TestGetQuotaReturnsExactLiveMeterAndFenceState(t *testing.T) {
	now := time.Unix(1_800_000_000, 123_000_000).UTC()
	remaining, overage := "0", "3"
	projection := testProjection()
	repository := &repositoryStub{snapshot: testSnapshot(projection)}
	meters := &meterStub{result: quotaruntime.MeterReadResult{AsOf: now, Meters: []quotaruntime.Meter{{
		PublicMeter: quota.PublicMeter{
			BindingID: "binding-1", RuleID: "rule-1", Metric: quota.MetricRequests,
			Enforcement: quota.EnforcementEnforce, Limit: "12", Used: "15", Remaining: &remaining,
			Overage: &overage, Completeness: quota.CompletenessPartial, KnownDispatches: "15",
			IncompleteDispatches: "2", CapacityState: quota.CapacityFenced,
		},
		Algorithm: quota.AlgorithmSlidingLog, Accounting: quota.AccountingRequest,
		ActiveFenceIDs: []string{"fence-b", "fence-a"},
	}}}}
	service := newTestService(t, repository, &appliedStub{policy: testAppliedPolicy(projection)}, meters, &waiterStub{})

	result, err := service.GetQuota(context.Background(), testNamespaceID, testKeySubject)
	if err != nil {
		t.Fatal(err)
	}
	if len(meters.requests) != 1 || meters.requests[0].Partition != "partition-1" || len(meters.requests[0].Rules) != 1 {
		t.Fatalf("unexpected exact meter read: %#v", meters.requests)
	}
	if len(result.Meters) != 1 || result.Meters[0].Meter.Used != "15" ||
		result.Meters[0].Meter.Remaining == nil || *result.Meters[0].Meter.Remaining != "0" ||
		result.Meters[0].Meter.Overage == nil || *result.Meters[0].Meter.Overage != "3" {
		t.Fatalf("live meter was not preserved: %#v", result.Meters)
	}
	if result.LimitingRuleID != "rule-1" || !result.AsOf.Equal(now) ||
		!reflect.DeepEqual(result.FenceIDs, []string{"fence-a", "fence-b"}) {
		t.Fatalf("unexpected limiting/fence state: %#v", result)
	}
}

func TestGetQuotaMatchesRuntimeSortedMetersByCounterIdentity(t *testing.T) {
	now := time.Unix(1_800_000_100, 0).UTC()
	projection := testProjection()
	limit, err := quota.ParseQuotaInteger("100")
	if err != nil {
		t.Fatal(err)
	}
	projection.RateBindings = []accessprojection.RateBinding{
		{
			BindingID: "binding-a", PolicyID: "policy-a",
			SubjectID: "33333333-3333-4333-8333-333333333333", Source: accesscontrol.InheritanceLayerTeam,
			Mode: accesscontrol.RateBindingHardCap,
			Rules: []accessprojection.ProjectedRateRule{
				{Rule: quota.RateLimitRule{
					ID: "rule-a0", Metric: quota.MetricRequests, Algorithm: quota.AlgorithmSlidingLog,
					Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
					WholeLimit: &limit, Window: time.Minute, Ordinal: 0,
				}},
				{Rule: quota.RateLimitRule{
					ID: "rule-a1", Metric: quota.MetricRequests, Algorithm: quota.AlgorithmSlidingLog,
					Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
					WholeLimit: &limit, Window: time.Hour, Ordinal: 1,
				}},
			},
		},
		{
			BindingID: "binding-b", PolicyID: "policy-b",
			SubjectID: testKeySubject.ID, Source: accesscontrol.InheritanceLayerKey,
			Mode: accesscontrol.RateBindingAllocation,
			Rules: []accessprojection.ProjectedRateRule{{Rule: quota.RateLimitRule{
				ID: "rule-b0", Metric: quota.MetricRequests, Algorithm: quota.AlgorithmSlidingLog,
				Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
				WholeLimit: &limit, Window: time.Minute, Ordinal: 0,
			}}},
		},
	}

	meter := func(bindingID, ruleID, used string) quotaruntime.Meter {
		remaining := "99"
		return quotaruntime.Meter{PublicMeter: quota.PublicMeter{
			BindingID: bindingID, RuleID: ruleID, Metric: quota.MetricRequests,
			Enforcement: quota.EnforcementEnforce, Limit: "100", Used: used,
			Remaining: &remaining, Completeness: quota.CompletenessComplete,
			KnownDispatches: used, IncompleteDispatches: "0", CapacityState: quota.CapacityAvailable,
		}, Algorithm: quota.AlgorithmSlidingLog, Accounting: quota.AccountingRequest}
	}
	// QuotaRuntime sorts globally by rule ordinal, then counter identity. That
	// differs from the projection's binding-major order whenever a binding has
	// more than one rule and another binding shares an earlier ordinal.
	meters := &meterStub{result: quotaruntime.MeterReadResult{AsOf: now, Meters: []quotaruntime.Meter{
		meter("binding-a", "rule-a0", "1"),
		meter("binding-b", "rule-b0", "2"),
		meter("binding-a", "rule-a1", "3"),
	}}}
	repository := &repositoryStub{snapshot: testSnapshot(projection)}
	service := newTestService(t, repository, &appliedStub{policy: testAppliedPolicy(projection)}, meters, &waiterStub{})

	result, err := service.GetQuota(context.Background(), testNamespaceID, testKeySubject)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Meters) != 3 ||
		result.Meters[0].Binding.BindingID != "binding-a" || result.Meters[0].Rule.Rule.ID != "rule-a0" || result.Meters[0].Meter.Used != "1" ||
		result.Meters[1].Binding.BindingID != "binding-a" || result.Meters[1].Rule.Rule.ID != "rule-a1" || result.Meters[1].Meter.Used != "3" ||
		result.Meters[2].Binding.BindingID != "binding-b" || result.Meters[2].Rule.Rule.ID != "rule-b0" || result.Meters[2].Meter.Used != "2" {
		t.Fatalf("quota meters were not joined by counter identity: %#v", result.Meters)
	}
}

func TestAccessCheckUsesStoredProjectionWithoutReadingOrConsumingQuota(t *testing.T) {
	projection := testProjection()
	repository := &repositoryStub{snapshot: testSnapshot(projection), exists: true}
	meters := &meterStub{err: errors.New("meter reads are forbidden for access simulation")}
	service := newTestService(t, repository, &appliedStub{policy: testAppliedPolicy(projection)}, meters, &waiterStub{})

	result, err := service.Check(context.Background(), AccessCheckRequest{
		NamespaceID: testNamespaceID, Subject: testKeySubject,
		Resource:   accesscontrol.GrantResource{Type: accesscontrol.GrantResourceModel, ID: "model-1"},
		Permission: accesscontrol.GrantPermissionInvoke,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != accesscontrol.AccessDecisionAllow || result.Simulation || len(result.Matched) != 1 {
		t.Fatalf("unexpected access decision: %#v", result)
	}
	if len(meters.requests) != 0 {
		t.Fatalf("access check touched quota meters: %#v", meters.requests)
	}
}

func TestAccessCheckMarksPrivilegedContextOverrideAsSimulation(t *testing.T) {
	projection := testProjection()
	snapshot := testSnapshot(projection)
	minimum, maximum := int64(0), int64(10)
	snapshot.Schema = RoutingClaimSchema{Revision: 3, Definitions: map[string]ClaimDefinition{
		"priority": {Kind: "integer", Minimum: &minimum, Maximum: &maximum},
	}}
	snapshot.Context.Effective = []EffectiveClaim{{StoredClaim: StoredClaim{
		Name:  "priority",
		Value: routingsnapshot.ClaimValue{Kind: "integer", Integer: 1},
	}, Source: *snapshot.LayerSubjects.Team}}
	repository := &repositoryStub{snapshot: snapshot, exists: true}
	service := newTestService(t, repository, &appliedStub{policy: testAppliedPolicy(projection)}, &meterStub{}, &waiterStub{})

	result, err := service.Check(context.Background(), AccessCheckRequest{
		NamespaceID: testNamespaceID, Subject: testKeySubject,
		Resource:   accesscontrol.GrantResource{Type: accesscontrol.GrantResourceModel, ID: "model-1"},
		Permission: accesscontrol.GrantPermissionInvoke, OverridePresent: true,
		Override: map[string]routingsnapshot.ClaimValue{"priority": {Kind: "integer", Integer: 8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Simulation || len(result.RoutingContext) != 1 || result.RoutingContext[0].Value.Integer != 8 ||
		result.RoutingContext[0].Source != testKeySubject {
		t.Fatalf("override was not isolated as simulation: %#v", result)
	}
}

func TestUpdateRoutingContextValidatesCASAndWaitsForPublication(t *testing.T) {
	projection := testProjection()
	snapshot := testSnapshot(projection)
	maximum := int64(64)
	snapshot.Schema = RoutingClaimSchema{Revision: 2, Definitions: map[string]ClaimDefinition{
		"segment": {Kind: "string", MaxLength: &maximum},
	}}
	snapshot.SubjectRevision = 11
	snapshot.Context.Revision = 11
	repository := &repositoryStub{snapshot: snapshot, mutation: RoutingContextMutation{
		DesiredRevision: 19, QuotaPartition: "partition-1",
	}}
	waiter := &waiterStub{}
	service := newTestService(t, repository, &appliedStub{}, &meterStub{}, waiter)
	values := map[string]routingsnapshot.ClaimValue{"segment": {Kind: "string", String: "research"}}

	_, err := service.UpdateRoutingContext(context.Background(), UpdateRoutingContextRequest{
		NamespaceID: testNamespaceID, Subject: testKeySubject, ExpectedRevision: 11, Values: values,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(repository.update.Values, values) || waiter.namespace != testNamespaceID ||
		waiter.partition != "partition-1" || waiter.revision != 19 {
		t.Fatalf("mutation/barrier contract was not preserved: update=%#v waiter=%#v", repository.update, waiter)
	}

	_, err = service.UpdateRoutingContext(context.Background(), UpdateRoutingContextRequest{
		NamespaceID: testNamespaceID, Subject: testKeySubject, ExpectedRevision: 10, Values: values,
	})
	if !errors.Is(err, ErrRevisionConflict) {
		t.Fatalf("stale CAS error = %v, want ErrRevisionConflict", err)
	}
}

const testNamespaceID = "11111111-1111-4111-8111-111111111111"

var testKeySubject = Subject{Kind: accesscontrol.SubjectKindAPIKey, ID: "22222222-2222-4222-8222-222222222222"}

func testProjection() accessprojection.Projection {
	limit, err := quota.ParseQuotaInteger("12")
	if err != nil {
		panic(err)
	}
	return accessprojection.Projection{
		NamespaceID: testNamespaceID, QuotaPartition: "partition-1", BillingCurrency: "USD",
		KeyID: testKeySubject.ID, Revision: 7, Digest: strings.Repeat("d", 64),
		Grants: []accessprojection.Grant{{
			BindingID: "access-binding-1", PolicyID: "access-policy-1",
			Source: accesscontrol.InheritanceLayerUser, ResourceType: accesscontrol.GrantResourceModel,
			ResourceID: "model-1", Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
		}},
		RateBindings: []accessprojection.RateBinding{
			{
				BindingID: "binding-1", PolicyID: "policy-1",
				SubjectID: "33333333-3333-4333-8333-333333333333", Source: accesscontrol.InheritanceLayerTeam,
				Mode: accesscontrol.RateBindingHardCap,
				Rules: []accessprojection.ProjectedRateRule{
					{Rule: quota.RateLimitRule{
						ID: "rule-1", Metric: quota.MetricRequests, Algorithm: quota.AlgorithmSlidingLog,
						Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
						WholeLimit: &limit, Window: time.Minute,
					}},
				},
			},
		},
	}
}

func testAppliedPolicy(projection accessprojection.Projection) accessruntime.AppliedPolicy {
	return accessruntime.AppliedPolicy{
		Active: accessruntime.ActivePolicy{
			KeyID: testKeySubject.ID, Revision: projection.Revision, Digest: projection.Digest,
			RoutingRevision: 9, RoutingSnapshotHash: strings.Repeat("e", 64),
		},
		Projection: projection,
	}
}

func testSnapshot(projection accessprojection.Projection) PolicySnapshot {
	user := Subject{Kind: accesscontrol.SubjectKindUser, ID: "44444444-4444-4444-8444-444444444444"}
	team := Subject{Kind: accesscontrol.SubjectKindTeam, ID: "33333333-3333-4333-8333-333333333333"}
	return PolicySnapshot{
		NamespaceID: testNamespaceID, QuotaPartition: "partition-1", Subject: testKeySubject,
		SubjectRevision: 11, DesiredRevision: 7, AppliedRevision: 7, Projection: projection,
		LayerSubjects: LayerSubjects{Key: &testKeySubject, User: &user, Team: &team},
		Schema:        RoutingClaimSchema{}, Context: RoutingContext{Subject: testKeySubject, Revision: 11},
	}
}

func newTestService(
	t *testing.T,
	repository Repository,
	applied AppliedPolicyReader,
	meters MeterReader,
	waiter PublicationWaiter,
) *Service {
	t.Helper()
	service, err := NewService(ServiceOptions{
		Repository: repository, Applied: applied, Routing: &routingSnapshotStub{}, Meters: meters, Waiter: waiter,
	})
	if err != nil {
		t.Fatal(err)
	}
	return service
}
