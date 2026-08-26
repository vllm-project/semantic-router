package managementcomposition

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type routingPublicationStoreStub struct {
	reference accesspublisher.NamespacePublication
	heads     []accesspublisher.PublicationHeads
	loaded    accesspublisher.LoadedRoutingPublication
	err       error
	headReads int
	loads     int
}

func (stub *routingPublicationStoreStub) GetPublicationNamespace(
	context.Context,
	string,
) (accesspublisher.NamespacePublication, error) {
	return stub.reference, stub.err
}

func (stub *routingPublicationStoreStub) ReadPublicationHeads(
	context.Context,
	accesspublisher.NamespacePublication,
) (accesspublisher.PublicationHeads, error) {
	if stub.err != nil {
		return accesspublisher.PublicationHeads{}, stub.err
	}
	if stub.headReads >= len(stub.heads) {
		return accesspublisher.PublicationHeads{}, errors.New("unexpected publication-head read")
	}
	value := stub.heads[stub.headReads]
	stub.headReads++
	return value, nil
}

func (stub *routingPublicationStoreStub) LoadRoutingPublication(
	context.Context,
	accesspublisher.RuntimePublicationIdentity,
) (accesspublisher.LoadedRoutingPublication, error) {
	stub.loads++
	return stub.loaded, stub.err
}

func TestAccessRoutingPublicationReaderUsesExactRuntimeDocumentPin(t *testing.T) {
	pin, identity, loaded := routingPublicationFixture(t)
	if loaded.Snapshot.Digest == pin.RoutingDocumentDigest {
		t.Fatal("fixture did not distinguish document and nested snapshot digests")
	}
	store := &routingPublicationStoreStub{
		reference: accesspublisher.NamespacePublication{
			NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
		},
		heads: []accesspublisher.PublicationHeads{
			{Active: &identity},
			{Active: &identity},
		},
		loaded: loaded,
	}
	reader := &accessRoutingPublicationReader{store: store}

	publication, err := reader.ReadRoutingPublication(context.Background(), pin)
	if err != nil {
		t.Fatal(err)
	}
	if publication.RoutingDocumentDigest != pin.RoutingDocumentDigest ||
		publication.Snapshot.Digest != loaded.Snapshot.Digest ||
		store.headReads != 2 || store.loads != 1 {
		t.Fatalf("exact runtime publication = %#v, head reads=%d, loads=%d", publication, store.headReads, store.loads)
	}
}

func TestAccessRoutingPublicationReaderRejectsPinMismatchWithoutLoading(t *testing.T) {
	pin, identity, loaded := routingPublicationFixture(t)
	pin.RoutingDocumentDigest = strings.Repeat("f", 64)
	store := &routingPublicationStoreStub{
		reference: accesspublisher.NamespacePublication{
			NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
		},
		heads:  []accesspublisher.PublicationHeads{{Active: &identity}},
		loaded: loaded,
	}
	reader := &accessRoutingPublicationReader{store: store}

	if _, err := reader.ReadRoutingPublication(context.Background(), pin); err == nil {
		t.Fatal("mismatched applied routing pin was accepted")
	}
	if store.loads != 0 {
		t.Fatalf("mismatched pin loaded %d publications", store.loads)
	}
}

func TestAccessRoutingPublicationReaderRejectsHeadAdvanceDuringLoad(t *testing.T) {
	pin, identity, loaded := routingPublicationFixture(t)
	newer := identity
	newer.PublicationID = "publication-8"
	newer.DesiredRevision++
	newer.RoutingDigest = strings.Repeat("e", 64)
	store := &routingPublicationStoreStub{
		reference: accesspublisher.NamespacePublication{
			NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
		},
		heads: []accesspublisher.PublicationHeads{
			{Active: &identity},
			{Active: &newer},
		},
		loaded: loaded,
	}
	reader := &accessRoutingPublicationReader{store: store}

	if _, err := reader.ReadRoutingPublication(context.Background(), pin); err == nil {
		t.Fatal("publication head advance returned a mixed catalog")
	}
}

func routingPublicationFixture(
	t *testing.T,
) (accessmanagement.RoutingPublicationPin, accesspublisher.RuntimePublicationIdentity, accesspublisher.LoadedRoutingPublication) {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "namespace-1", Revision: 7, Currency: "USD",
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := accesspublisher.RuntimePublicationIdentity{
		PublicationID: "publication-7", NamespaceID: snapshot.NamespaceID,
		QuotaPartition: "partition-1", DesiredRevision: 7, RuntimeEpoch: 2,
		PublicationDigest: strings.Repeat("a", 64), ManifestDigest: strings.Repeat("b", 64),
		RoutingDigest: strings.Repeat("d", 64), State: accesspublisher.PublicationStateActive,
	}
	pin := accessmanagement.RoutingPublicationPin{
		NamespaceID: identity.NamespaceID, QuotaPartition: identity.QuotaPartition,
		PublicationID: identity.PublicationID, RuntimeEpoch: identity.RuntimeEpoch,
		RoutingRevision: snapshot.Revision, RoutingDocumentDigest: identity.RoutingDigest,
	}
	return pin, identity, accesspublisher.LoadedRoutingPublication{
		Identity: identity, Snapshot: *snapshot,
	}
}
