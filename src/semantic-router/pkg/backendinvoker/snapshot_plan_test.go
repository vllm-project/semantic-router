package backendinvoker

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type snapshotSourceStub struct {
	snapshot      *routingsnapshot.Snapshot
	routingDigest string
	err           error
	calls         *int
}

func (s snapshotSourceStub) Snapshot(_ context.Context, pin routingcontext.Generation) (*routingsnapshot.Snapshot, error) {
	if s.calls != nil {
		(*s.calls)++
	}
	digest := s.routingDigest
	if digest == "" && s.snapshot != nil {
		digest = s.snapshot.Digest
	}
	if s.snapshot == nil || pin.NamespaceID != s.snapshot.NamespaceID || pin.SnapshotRevision != s.snapshot.Revision ||
		pin.RoutingDigest != digest {
		return nil, errors.New("routing snapshot pin does not match")
	}
	return s.snapshot, s.err
}

func TestSnapshotPlanResolverPinsExecutionAndPhysicalBackends(t *testing.T) {
	snapshot := compiledPlanSnapshot(t)
	resolver := &SnapshotPlanResolver{Source: snapshotSourceStub{snapshot: snapshot}}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: 9, AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{{
			DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 2,
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1", ModelRevision: 3,
		}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
	})
	capability.RoutingDigest = snapshot.Digest
	plans, err := resolver.ResolvePlans(context.Background(), capability)
	if err != nil {
		t.Fatalf("ResolvePlans() error = %v", err)
	}
	plan := plans.Candidates[0]
	if plan.Execution.MaxRetries != 2 || plan.Execution.RequestTimeout != 45*time.Second || plan.Execution.StreamTimeout != 10*time.Minute {
		t.Fatalf("execution = %+v", plan.Execution)
	}
	if len(plan.Backends) != 2 || plan.Backends[0].Weight != 1_250_000_000 || plan.Backends[1].Weight != 2_000_000_000 {
		t.Fatalf("backends = %+v", plan.Backends)
	}
	if plan.Backends[0].ProviderCredentialID != "credential-1" {
		t.Fatalf("provider credential not pinned: %+v", plan.Backends[0])
	}
	if plan.Backends[0].WireFormat != "openai.chat.v1" {
		t.Fatalf("wire format not pinned: %+v", plan.Backends[0])
	}
	if plan.Backends[0].Connection.Path != "/chat/completions" || plan.Backends[0].Connection.Headers.Get("X-Wire-Version") != "2026-08-22" {
		t.Fatalf("compiled wire connection not pinned: %+v", plan.Backends[0].Connection)
	}
}

func TestSnapshotPlanResolverNeverFallsBackAcrossModelRevision(t *testing.T) {
	snapshot := compiledPlanSnapshot(t)
	resolver := &SnapshotPlanResolver{Source: snapshotSourceStub{snapshot: snapshot}}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: 9, AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{{
			DispatchID: "dispatch-1", DispatchType: "primary",
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1", ModelRevision: 4,
		}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
	})
	capability.RoutingDigest = snapshot.Digest
	_, err := resolver.ResolvePlans(context.Background(), capability)
	if err == nil {
		t.Fatal("missing pinned model revision unexpectedly fell back")
	}
}

func TestSnapshotPlanResolverRejectsPinnedRoutingDigestMismatch(t *testing.T) {
	snapshot := compiledPlanSnapshot(t)
	resolver := &SnapshotPlanResolver{Source: snapshotSourceStub{snapshot: snapshot}}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: snapshot.Revision,
		AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{{
			DispatchID: "dispatch-1", DispatchType: "primary",
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1", ModelRevision: 3,
		}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
	})
	capability.RoutingDigest = strings.Repeat("f", 64)
	if _, err := resolver.ResolvePlans(context.Background(), capability); err == nil {
		t.Fatal("snapshot with a different pinned digest unexpectedly resolved")
	}
}

func TestSnapshotPlanResolverResolvesWholeChainFromOneExactSnapshot(t *testing.T) {
	snapshot := compiledPlanSnapshot(t)
	calls := 0
	resolver := &SnapshotPlanResolver{Source: snapshotSourceStub{snapshot: snapshot, calls: &calls}}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: snapshot.Revision,
		AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{
			{DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 0, DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1", ModelRevision: 3},
			{DispatchID: "dispatch-2", DispatchType: "primary", Ordinal: 1, DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-2", ModelRevision: 5, Priority: 1},
		},
		Fallback:      FallbackPolicy{On: []FallbackTrigger{FallbackUnavailable}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
	})
	capability.RoutingDigest = snapshot.Digest
	plans, err := resolver.ResolvePlans(context.Background(), capability)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(plans.Candidates) != 2 || plans.Candidates[1].ModelID != "model-2" ||
		plans.Candidates[1].ModelRevision != 5 || plans.Candidates[1].Priority != 1 {
		t.Fatalf("snapshot calls=%d plans=%+v", calls, plans)
	}

	capability.Candidates[1].ModelRevision++
	if partial, err := resolver.ResolvePlans(context.Background(), capability); err == nil || len(partial.Candidates) != 0 {
		t.Fatalf("mismatched second candidate returned partial plans=%+v error=%v", partial, err)
	}
}

func TestRequestUsesStreamingReadsCanonicalFlag(t *testing.T) {
	if !requestUsesStreaming([]byte(`{"stream":true}`)) {
		t.Fatal("stream=true was not detected")
	}
	if requestUsesStreaming([]byte(`{"stream":false}`)) || requestUsesStreaming([]byte(`not-json`)) {
		t.Fatal("non-streaming request was misclassified")
	}
}

func compiledPlanSnapshot(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	document, _ := json.Marshal(map[string]any{"kind": "recipe"})
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "namespace-1", Revision: 9,
		Models: []routingsnapshot.Model{
			{
				ID: "model-1", Revision: 3,
				CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				Name:            "remote/frontier",
				Execution:       routingsnapshot.ModelExecution{MaxRetries: 2, RequestTimeout: "45s", StreamTimeout: "10m"},
				Backends: []routingsnapshot.Backend{
					{ID: "backend-a", ProviderID: "provider-openai", WireFormat: "openai.chat.v1", Origin: "https://one.example/v1", ProviderModelID: "provider-a", ProviderCredentialID: "credential-1", Connection: routingsnapshot.BackendConnection{Path: "/chat/completions", Headers: map[string]string{"X-Wire-Version": "2026-08-22"}}, Weight: "1.25"},
					{ID: "backend-b", ProviderID: "provider-openai", WireFormat: "openai.chat.v1", Origin: "https://two.example/v1", ProviderModelID: "provider-b", Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "2"},
				},
			},
			{
				ID: "model-2", Revision: 5,
				CatalogRevision: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
				Name:            "remote/fallback",
				Execution:       routingsnapshot.ModelExecution{MaxRetries: 1, RequestTimeout: "30s", StreamTimeout: "5m"},
				Backends:        []routingsnapshot.Backend{{ID: "backend-c", ProviderID: "provider-openai", WireFormat: "openai.chat.v1", Origin: "https://three.example/v1", ProviderModelID: "provider-c", Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "1"}},
			},
		},
		Recipes: []routingsnapshot.Recipe{{ID: "recipe-1", Revision: 1, Name: "balance", Decisions: []routingsnapshot.Decision{{ID: "decision-1", Name: "Simple", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}}, Document: document}},
		Entrypoints: []routingsnapshot.Entrypoint{{ID: "entrypoint-1", Revision: 1, Name: "blend", Aliases: []string{"vllm-sr/blend"}, Rules: []routingsnapshot.EntrypointRule{{
			ID: "rule-1", Name: "default", RecipeID: "recipe-1", RecipeRevision: 1,
			Assignments: map[string]routingsnapshot.AssignmentSet{"decision-1": {Models: []routingsnapshot.Assignment{{ModelID: "model-1", ModelRevision: 3, Weight: "1"}}}},
		}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
