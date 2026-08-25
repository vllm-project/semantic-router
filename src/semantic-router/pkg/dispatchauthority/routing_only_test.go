package dispatchauthority

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const routingOnlyNamespaceID = "11111111-1111-4111-8111-111111111111"

type routingOnlyPublicationReader struct {
	identity accesspublisher.RuntimePublicationIdentity
	ready    bool
}

func (reader *routingOnlyPublicationReader) CurrentRoutingPublication(
	namespaceID string,
) (accesspublisher.RuntimePublicationIdentity, bool) {
	if reader == nil || !reader.ready || reader.identity.NamespaceID != namespaceID {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	return reader.identity, true
}

type routingOnlySnapshotSource struct {
	snapshot *routingsnapshot.Snapshot
}

func (source routingOnlySnapshotSource) Snapshot(
	_ context.Context,
	pin routingcontext.Generation,
) (*routingsnapshot.Snapshot, error) {
	if source.snapshot == nil || pin != routingOnlyGeneration(routingOnlyPublication(source.snapshot)) {
		return nil, nil
	}
	return source.snapshot, nil
}

func TestRoutingOnlyAuthorityPinsActiveGenerationModelAndRequest(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	snapshot := routingOnlySnapshot(t)
	publication := routingOnlyPublication(snapshot)
	reader := &routingOnlyPublicationReader{identity: publication, ready: true}
	options := testIssuerOptions(now)
	verificationKeyring := cloneTestKeyring(options.Keyring)
	authority, testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr := NewRoutingOnlyAuthority(RoutingOnlyAuthorityOptions{
		NamespaceID:  routingOnlyNamespaceID,
		Publications: reader,
		Issuer:       options,
	})
	if testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr != nil {
		t.Fatal(testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr)
	}
	defer authority.Close()
	if err := authority.AttachRoutingSnapshots(routingOnlySnapshotSource{snapshot: snapshot}); err != nil {
		t.Fatal(err)
	}

	generation := routingOnlyGeneration(publication)
	facts := accessruntime.DispatchFacts{
		DispatchID: "dispatch-1", Ordinal: 2,
		DispatchPlanDigest: strings.Repeat("c", 64),
	}
	request := RoutingOnlyIssueRequest{
		Generation: generation,
		Dispatch:   facts,
		RequestID:  "request-1",
		Final: FinalRequest{
			Model:  ModelIdentity{ID: "model-1", Revision: 3},
			Method: "POST", Path: "/v1/chat/completions", WireFormat: "openai.chat.v1", Body: []byte(`{"model":"public/blend"}`),
		},
	}
	token, testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr := authority.IssuePrimary(context.Background(), request)
	if testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr != nil {
		t.Fatal(testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr)
	}
	capability, testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr := verificationKeyring.Verify(token, options.Audience, now)
	if testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr != nil {
		t.Fatal(testRoutingOnlyAuthorityPinsActiveGenerationModelAndRequestErr)
	}
	if len(capability.Candidates) != 1 {
		t.Fatalf("capability candidates = %+v", capability.Candidates)
	}
	candidate := capability.Candidates[0]
	if capability.NamespaceID != generation.NamespaceID ||
		capability.PublicationID != generation.PublicationID ||
		capability.RuntimeEpoch != generation.RuntimeEpoch ||
		capability.RoutingRevision != generation.SnapshotRevision ||
		capability.RoutingDigest != generation.RoutingDigest ||
		capability.RequestID != request.RequestID ||
		candidate.ModelID != request.Final.Model.ID ||
		candidate.ModelRevision != request.Final.Model.Revision ||
		!strings.HasPrefix(capability.AdmissionID, "public-") ||
		capability.AdmissionDigest == "" {
		t.Fatalf("capability = %+v", capability)
	}

	request.Final.Model.Revision++
	if _, err := authority.IssuePrimary(context.Background(), request); err == nil {
		t.Fatal("IssuePrimary() accepted a model outside the active snapshot")
	}
	request.Final.Model.Revision--
	request.Generation.RuntimeEpoch++
	if _, err := authority.IssuePrimary(context.Background(), request); err == nil {
		t.Fatal("IssuePrimary() accepted a stale publication generation")
	}
}

func TestRoutingOnlyGrantRechecksRequestAndActiveGeneration(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	snapshot := routingOnlySnapshot(t)
	publication := routingOnlyPublication(snapshot)
	reader := &routingOnlyPublicationReader{identity: publication, ready: true}
	authority, testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr := NewRoutingOnlyAuthority(RoutingOnlyAuthorityOptions{
		NamespaceID:  routingOnlyNamespaceID,
		Publications: reader,
		Issuer:       testIssuerOptions(now),
	})
	if testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr != nil {
		t.Fatal(testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr)
	}
	defer authority.Close()
	if err := authority.AttachRoutingSnapshots(routingOnlySnapshotSource{snapshot: snapshot}); err != nil {
		t.Fatal(err)
	}
	generation := routingOnlyGeneration(publication)
	grant, testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr := authority.IssueGrant(context.Background(), RoutingOnlyGrantIssueRequest{
		Generation: generation,
		Dispatch: accessruntime.DispatchFacts{
			DispatchID: "dispatch-1", DispatchPlanDigest: strings.Repeat("c", 64),
		},
		RequestID: "request-1",
		Model:     ModelIdentity{ID: "model-1", Revision: 3},
	})
	if testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr != nil {
		t.Fatal(testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr)
	}
	if _, err := authority.VerifyGrant(context.Background(), grant, GrantVerificationRequest{
		Generation: generation, RequestID: "another-request",
	}); err == nil {
		t.Fatal("VerifyGrant() accepted a different request")
	}
	verified, testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr := authority.VerifyGrant(context.Background(), grant, GrantVerificationRequest{
		Generation: generation, RequestID: "request-1",
	})
	if testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr != nil {
		t.Fatal(testRoutingOnlyGrantRechecksRequestAndActiveGenerationErr)
	}
	final := FinalRequest{
		Model:  ModelIdentity{ID: "model-1", Revision: 3},
		Method: "POST", Path: "/v1/chat/completions", WireFormat: "openai.chat.v1", Body: []byte(`{"nested":true}`),
	}
	if _, err := authority.IssueFromGrant(context.Background(), verified, final); err != nil {
		t.Fatal(err)
	}
	reader.identity.RuntimeEpoch++
	if _, err := authority.IssueFromGrant(context.Background(), verified, final); err == nil {
		t.Fatal("IssueFromGrant() accepted a no-longer-active generation")
	}
}

func TestRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshot(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	snapshot := routingOnlySnapshot(t)
	publication := routingOnlyPublication(snapshot)
	reader := &routingOnlyPublicationReader{identity: publication, ready: true}
	options := testIssuerOptions(now)
	verificationKeyring := cloneTestKeyring(options.Keyring)
	authority, testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr := NewRoutingOnlyAuthority(RoutingOnlyAuthorityOptions{
		NamespaceID: routingOnlyNamespaceID, Publications: reader, Issuer: options,
	})
	if testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr != nil {
		t.Fatal(testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr)
	}
	defer authority.Close()
	if err := authority.AttachRoutingSnapshots(routingOnlySnapshotSource{snapshot: snapshot}); err != nil {
		t.Fatal(err)
	}
	token, testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr := authority.IssueChain(context.Background(), RoutingOnlyChainIssueRequest{
		Generation: routingOnlyGeneration(publication),
		Candidates: []CandidateIssue{
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-0", DispatchPlanDigest: strings.Repeat("a", 64)}, Model: ModelIdentity{ID: "model-1", Revision: 3}},
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-1", Ordinal: 1, DispatchPlanDigest: strings.Repeat("c", 64)}, Model: ModelIdentity{ID: "model-2", Revision: 4}, Priority: 1},
		},
		Fallback:  backendinvoker.FallbackPolicy{On: []backendinvoker.FallbackTrigger{backendinvoker.FallbackUnavailable}},
		RequestID: "request-1", Final: ChainFinalRequest{Method: "POST", Path: "/v1/chat/completions", WireFormat: "openai.chat.v1", Body: []byte(`{}`)},
	})
	if testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr != nil {
		t.Fatal(testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr)
	}
	capability, testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr := verificationKeyring.Verify(token, options.Audience, now)
	if testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr != nil || len(capability.Candidates) != 2 || capability.Candidates[1].ModelID != "model-2" {
		t.Fatalf("capability = %+v, %v", capability, testRoutingOnlyAuthorityIssuesChainOnlyFromOneActiveSnapshotErr)
	}
	reader.identity.RuntimeEpoch++
	if _, err := authority.IssueChain(context.Background(), RoutingOnlyChainIssueRequest{
		Generation: routingOnlyGeneration(publication),
		Candidates: []CandidateIssue{{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-0", DispatchPlanDigest: strings.Repeat("a", 64)}, Model: ModelIdentity{ID: "model-1", Revision: 3}}},
		RequestID:  "request-1", Final: ChainFinalRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(`{}`)},
	}); err == nil {
		t.Fatal("chain from a stale generation unexpectedly issued")
	}
}

func TestRoutingOnlyAuthorityConstructionAttachmentAndCloseFailClosed(t *testing.T) {
	if _, err := NewRoutingOnlyAuthority(RoutingOnlyAuthorityOptions{}); err == nil {
		t.Fatal("NewRoutingOnlyAuthority() accepted an empty public namespace")
	}
	now := time.Unix(1_900_000_000, 0).UTC()
	snapshot := routingOnlySnapshot(t)
	reader := &routingOnlyPublicationReader{identity: routingOnlyPublication(snapshot), ready: true}
	authority, err := NewRoutingOnlyAuthority(RoutingOnlyAuthorityOptions{
		NamespaceID:  routingOnlyNamespaceID,
		Publications: reader,
		Issuer:       testIssuerOptions(now),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := authority.AttachRoutingSnapshots(routingOnlySnapshotSource{snapshot: snapshot}); err != nil {
		t.Fatal(err)
	}
	if err := authority.AttachRoutingSnapshots(routingOnlySnapshotSource{snapshot: snapshot}); err == nil {
		t.Fatal("AttachRoutingSnapshots() accepted a replacement source")
	}
	if err := authority.Close(); err != nil {
		t.Fatal(err)
	}
	if err := authority.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
	if _, err := authority.IssuePrimary(context.Background(), RoutingOnlyIssueRequest{}); err == nil {
		t.Fatal("closed authority issued a capability")
	}
}

func routingOnlySnapshot(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: routingOnlyNamespaceID,
		Revision:    29,
		Models: []routingsnapshot.Model{{
			ID: "model-1", Revision: 3,
			CatalogRevision: "sha256:" + strings.Repeat("a", 64),
			Name:            "remote/frontier",
			Execution: routingsnapshot.ModelExecution{
				MaxRetries: 1, RequestTimeout: "30s", StreamTimeout: "5m",
			},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-1", ProviderID: "provider-1", WireFormat: "openai.chat.v1",
				Origin: "https://models.example", ProviderModelID: "provider-model",
				Connection: routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}, {
			ID: "model-2", Revision: 4,
			CatalogRevision: "sha256:" + strings.Repeat("b", 64), Name: "remote/fallback",
			Execution: routingsnapshot.ModelExecution{MaxRetries: 0, RequestTimeout: "30s", StreamTimeout: "5m"},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-2", ProviderID: "provider-2", WireFormat: "openai.chat.v1",
				Origin: "https://fallback.example", ProviderModelID: "provider-fallback",
				Connection: routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func routingOnlyPublication(snapshot *routingsnapshot.Snapshot) accesspublisher.RuntimePublicationIdentity {
	if snapshot.Revision <= 0 {
		panic("routing-only fixture requires a positive revision")
	}
	// #nosec G115 -- the fixture revision is checked positive above.
	desiredRevision := uint64(snapshot.Revision)
	return accesspublisher.RuntimePublicationIdentity{
		PublicationID: "publication-1", NamespaceID: snapshot.NamespaceID,
		QuotaPartition: "partition-1", DesiredRevision: desiredRevision, RuntimeEpoch: 2,
		PublicationDigest: strings.Repeat("1", 64), ManifestDigest: strings.Repeat("2", 64),
		RoutingDigest: snapshot.Digest, State: accesspublisher.PublicationStateActive,
	}
}

func routingOnlyGeneration(publication accesspublisher.RuntimePublicationIdentity) routingcontext.Generation {
	if publication.DesiredRevision > math.MaxInt64 {
		panic("routing-only fixture revision exceeds int64")
	}
	// #nosec G115 -- the fixture revision is bounded to MaxInt64 above.
	snapshotRevision := int64(publication.DesiredRevision)
	return routingcontext.Generation{
		NamespaceID: publication.NamespaceID, QuotaPartition: publication.QuotaPartition,
		PublicationID: publication.PublicationID, RuntimeEpoch: publication.RuntimeEpoch,
		SnapshotRevision: snapshotRevision, RoutingDigest: publication.RoutingDigest,
	}
}
