package extproc

import (
	"context"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type dispatchCapabilityRuntimeStub struct{ metered bool }

func (stub dispatchCapabilityRuntimeStub) Metered() bool { return stub.metered }

func (dispatchCapabilityRuntimeStub) IssueMeteredPrimary(dispatchauthority.PrimaryIssueRequest) (string, error) {
	return "capability", nil
}

func (dispatchCapabilityRuntimeStub) IssueMeteredGrant(dispatchauthority.GrantIssueRequest) (string, error) {
	return "grant", nil
}

func (dispatchCapabilityRuntimeStub) IssueMeteredChain(dispatchauthority.MeteredChainIssueRequest) (string, error) {
	return "capability", nil
}

func (dispatchCapabilityRuntimeStub) IssueRoutingOnlyPrimary(context.Context, dispatchauthority.RoutingOnlyIssueRequest) (string, error) {
	return "capability", nil
}

func (dispatchCapabilityRuntimeStub) IssueRoutingOnlyGrant(context.Context, dispatchauthority.RoutingOnlyGrantIssueRequest) (string, error) {
	return "grant", nil
}

func (dispatchCapabilityRuntimeStub) IssueRoutingOnlyChain(context.Context, dispatchauthority.RoutingOnlyChainIssueRequest) (string, error) {
	return "capability", nil
}

func (dispatchCapabilityRuntimeStub) VerifyGrant(context.Context, string, dispatchauthority.GrantVerificationRequest) (dispatchauthority.VerifiedGrant, error) {
	return dispatchauthority.VerifiedGrant{}, nil
}

func (dispatchCapabilityRuntimeStub) IssueFromGrant(context.Context, dispatchauthority.VerifiedGrant, dispatchauthority.FinalRequest) (string, error) {
	return "capability", nil
}

func (dispatchCapabilityRuntimeStub) VerifyDispatchOutcome(context.Context, string, dispatchauthority.OutcomeVerificationRequest) (backendinvoker.DispatchOutcome, error) {
	return backendinvoker.DispatchOutcome{}, nil
}

type routingPublicationReaderStub struct {
	identity accesspublisher.RuntimePublicationIdentity
	ready    bool
}

func (reader *routingPublicationReaderStub) CurrentRoutingPublication(
	namespaceID string,
) (accesspublisher.RuntimePublicationIdentity, bool) {
	if reader == nil || !reader.ready || reader.identity.NamespaceID != namespaceID {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	return reader.identity, true
}

func TestDurableRoutingRequestRuntimeAuthenticatesAndAcquiresExactPublication(t *testing.T) {
	registry, _ := newDurableRoutingRegistryForTest(t)
	publication := durableRoutingPublicationState(
		durableRoutingPublication(t, "namespace-a", "partition-a", 3, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), publication); err != nil {
		t.Fatal(err)
	}
	tenant := inferenceTestTenant("")
	tenant.NamespaceID = publication.Identity.NamespaceID
	tenant.QuotaPartition = publication.Identity.QuotaPartition
	tenant.PublicationID = publication.Identity.PublicationID
	tenant.RuntimeEpoch = publication.Identity.RuntimeEpoch
	tenant.RoutingRevision = publication.Snapshot.Revision
	tenant.RoutingDigest = publication.Identity.RoutingDigest
	access := &fakeInferenceAccess{authenticate: func(request accessruntime.AuthenticationRequest) (accessruntime.Authentication, error) {
		return accessruntime.Authentication{
			Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionAllowed},
			Tenant: tenant,
		}, nil
	}}
	publications := &routingPublicationReaderStub{identity: publication.Identity, ready: true}
	runtime, err := NewDurableRoutingRequestRuntime(DurableRoutingRequestRuntimeOptions{
		Access: access, Publications: publications, Routers: registry,
		Dispatch: dispatchCapabilityRuntimeStub{metered: true},
	})
	if err != nil {
		t.Fatal(err)
	}

	resolution, result, err := runtime.resolveExternal(context.Background(), "one-time-presented-secret")
	if err != nil || !result.Allowed() {
		t.Fatalf("resolveExternal() = (%+v, %v)", result, err)
	}
	if resolution.lease == nil || resolution.lease.Router == nil ||
		resolution.generation.PublicationID != publication.Identity.PublicationID {
		t.Fatalf("resolution = %+v", resolution)
	}
	if len(access.authentications) != 1 || access.authentications[0].Credential != "one-time-presented-secret" {
		t.Fatalf("authentication requests = %+v", access.authentications)
	}
	resolution.lease.Release()

	publications.identity.RuntimeEpoch++
	failed, result, err := runtime.resolveExternal(context.Background(), "second-secret")
	if err != nil || result.Disposition != quotaruntime.AdmissionUnavailable || failed.lease != nil {
		t.Fatalf("publication mismatch = (%+v, %+v, %v)", failed, result, err)
	}
}

func TestDurableRoutingRequestRuntimeRequiresEveryDependency(t *testing.T) {
	if _, err := NewDurableRoutingRequestRuntime(DurableRoutingRequestRuntimeOptions{}); err == nil {
		t.Fatal("empty durable routing request runtime was accepted")
	}
}

func TestDurableRoutingRoutingOnlyAcquiresConfiguredPublicNamespace(t *testing.T) {
	registry, _ := newDurableRoutingRegistryForTest(t)
	publication := durableRoutingPublicationState(
		durableRoutingPublication(t, "11111111-1111-4111-8111-111111111111", "partition-public", 3, 7, "p"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), publication); err != nil {
		t.Fatal(err)
	}
	publications := &routingPublicationReaderStub{identity: publication.Identity, ready: true}
	runtime, err := NewDurableRoutingRequestRuntime(DurableRoutingRequestRuntimeOptions{
		PublicNamespaceID: publication.Identity.NamespaceID,
		Publications:      publications, Routers: registry, Dispatch: dispatchCapabilityRuntimeStub{},
	})
	if err != nil {
		t.Fatal(err)
	}
	resolution, err := runtime.resolvePublic()
	if err != nil || resolution.lease == nil || resolution.authentication.Tenant.NamespaceID != "" {
		t.Fatalf("resolvePublic() = %+v, %v", resolution, err)
	}
	if resolution.generation.NamespaceID != publication.Identity.NamespaceID {
		t.Fatalf("public generation = %+v", resolution.generation)
	}
	resolution.lease.Release()
}

func TestConsumeBearerCredentialRejectsDuplicatesAndErasesRawHeaders(t *testing.T) {
	headers := &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: "Authorization", Value: "Bearer first"},
		{Key: "authorization", RawValue: []byte("Bearer second")},
	}}
	if credential, ok := consumeBearerCredential(headers); ok || credential != "" {
		t.Fatalf("duplicate bearer = %q, %v", credential, ok)
	}
	if len(headers.Headers) != 1 || headers.Headers[0].Key != ":method" {
		t.Fatalf("sanitized headers = %+v", headers.Headers)
	}

	headers = &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: "Authorization", Value: "Bearer accepted"},
	}}
	credential, ok := consumeBearerCredential(headers)
	if !ok || credential != "accepted" || len(headers.Headers) != 1 {
		t.Fatalf("valid bearer = %q, %v, headers=%+v", credential, ok, headers.Headers)
	}
}
