package delegationmanagement

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
)

const publicationWaiterRoutingDigest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestPublicationTransitionRetriesOnlyAbsentProjection(t *testing.T) {
	ready, err := publicationTransition(accessruntime.ErrProjectionNotFound)
	if ready || err != nil {
		t.Fatalf("missing projection transition = ready:%v err:%v", ready, err)
	}
	ready, err = publicationTransition(accessruntime.ErrPublicationPending)
	if ready || err != nil {
		t.Fatalf("pending publication transition = ready:%v err:%v", ready, err)
	}

	runtimeErr := errors.New("runtime read failed")
	ready, err = publicationTransition(runtimeErr)
	if ready || !errors.Is(err, runtimeErr) {
		t.Fatalf("runtime failure transition = ready:%v err:%v", ready, err)
	}
}

func TestAPIKeyPublicationWaiterRetriesAProvablyNewerPublication(t *testing.T) {
	projection := publicationWaiterProjection(t)
	reader := &publicationRaceReader{advance: true, projection: projection}
	waiter := &RedisPublicationWaiter{reader: reader, store: &publicationWaiterStore{}}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := waiter.WaitAPIKeyActive(ctx, projection.NamespaceID, projection.KeyID, "public-key"); err != nil {
		t.Fatalf("WaitAPIKeyActive() = %v", err)
	}
	if reader.locates < 3 {
		t.Fatalf("location reads = %d, want transition check and retry", reader.locates)
	}
}

func TestAPIKeyPublicationWaiterRejectsCorruptionUnderSamePublication(t *testing.T) {
	projection := publicationWaiterProjection(t)
	waiter := &RedisPublicationWaiter{
		reader: &publicationRaceReader{projection: projection}, store: &publicationWaiterStore{},
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	err := waiter.WaitAPIKeyActive(ctx, projection.NamespaceID, projection.KeyID, "public-key")
	if !errors.Is(err, accessruntime.ErrRuntimeCorrupt) {
		t.Fatalf("WaitAPIKeyActive() error = %v, want corrupt active publication", err)
	}
}

func TestAPIKeyPublicationWaiterWaitsForEveryActiveReplica(t *testing.T) {
	projection := publicationWaiterProjection(t)
	store := &publicationWaiterStore{missingCalls: 2}
	waiter := &RedisPublicationWaiter{
		reader: &publicationReadyReader{projection: projection}, store: store,
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := waiter.WaitAPIKeyActive(ctx, projection.NamespaceID, projection.KeyID, "public-key"); err != nil {
		t.Fatalf("WaitAPIKeyActive() = %v", err)
	}
	if store.activeCalls < 3 {
		t.Fatalf("active replica checks = %d, want at least 3", store.activeCalls)
	}
}

func TestAppliedPublicationWaiterWaitsForEveryActiveReplica(t *testing.T) {
	store := &publicationWaiterStore{missingCalls: 2}
	waiter := &RedisPublicationWaiter{store: store}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := waiter.WaitApplied(ctx, "namespace-1", "partition-1", 7); err != nil {
		t.Fatalf("WaitApplied() = %v", err)
	}
	if store.activeCalls < 3 {
		t.Fatalf("active replica checks = %d, want at least 3", store.activeCalls)
	}
	want := (accesspublisher.ActiveGeneration{
		PublicationID: "publication-7", Revision: 7, RuntimeEpoch: 3,
		RoutingSnapshotDigest: publicationWaiterRoutingDigest,
	})
	if store.lastGeneration != want {
		t.Fatalf("active generation = %+v, want %+v", store.lastGeneration, want)
	}
}

type publicationWaiterStore struct {
	missingCalls   int
	activeCalls    int
	lastGeneration accesspublisher.ActiveGeneration
}

func (*publicationWaiterStore) Readiness(
	context.Context,
	string,
	string,
) (accesspublisher.Readiness, error) {
	return accesspublisher.Readiness{
		Ready: true, RuntimeEpoch: 3, DesiredRevision: 7, AppliedRevision: 7,
		AccessGate: "publication-7", RoutingGate: "publication-7", RoutingDigest: publicationWaiterRoutingDigest,
	}, nil
}

func (store *publicationWaiterStore) ActiveReplicaAcknowledgements(
	_ context.Context,
	_ string,
	_ string,
	generation accesspublisher.ActiveGeneration,
) (accesspublisher.ActiveReplicaStatus, error) {
	store.activeCalls++
	store.lastGeneration = generation
	if store.activeCalls <= store.missingCalls {
		return accesspublisher.ActiveReplicaStatus{Required: []string{"router-a"}, Missing: []string{"router-a"}}, nil
	}
	return accesspublisher.ActiveReplicaStatus{Required: []string{"router-a"}}, nil
}

type publicationReadyReader struct {
	projection accessprojection.Projection
}

func (reader *publicationReadyReader) LocateCredential(
	context.Context,
	accesscredential.Kind,
	string,
) (accessruntime.CredentialLocation, error) {
	return accessruntime.CredentialLocation{
		NamespaceID: reader.projection.NamespaceID, QuotaPartition: reader.projection.QuotaPartition,
		PublicationID: "publication-1", RuntimeEpoch: 1, RoutingRevision: 1,
		RoutingDocumentDigest: publicationWaiterRoutingDigest,
	}, nil
}

func (reader *publicationReadyReader) LocateCredentialCoherent(
	ctx context.Context,
	kind accesscredential.Kind,
	publicID string,
) (accessruntime.CredentialLocation, error) {
	return reader.LocateCredential(ctx, kind, publicID)
}

func (reader *publicationReadyReader) ReadCredential(
	context.Context,
	accessruntime.CredentialLocation,
	accesscredential.Kind,
	string,
) (accessprojection.CredentialProjection, error) {
	return accessprojection.CredentialProjection{KeyID: reader.projection.KeyID}, nil
}

func (reader *publicationReadyReader) ReadActivePolicy(
	_ context.Context,
	location accessruntime.CredentialLocation,
	keyID string,
) (accessruntime.ActivePolicy, error) {
	return accessruntime.ActivePolicy{
		KeyID: keyID, Revision: reader.projection.Revision, Digest: reader.projection.Digest,
		PublicationID: location.PublicationID, RuntimeEpoch: location.RuntimeEpoch,
		RoutingRevision: location.RoutingRevision, RoutingDocumentDigest: location.RoutingDocumentDigest,
	}, nil
}

func (reader *publicationReadyReader) ReadPolicy(
	context.Context,
	accessruntime.CredentialLocation,
	accessruntime.ActivePolicy,
) (accessprojection.Projection, error) {
	return reader.projection, nil
}

type publicationRaceReader struct {
	advance    bool
	locates    int
	projection accessprojection.Projection
}

func (reader *publicationRaceReader) LocateCredential(
	context.Context,
	accesscredential.Kind,
	string,
) (accessruntime.CredentialLocation, error) {
	reader.locates++
	location := accessruntime.CredentialLocation{
		NamespaceID: reader.projection.NamespaceID, QuotaPartition: reader.projection.QuotaPartition,
		PublicationID: "publication-1", RuntimeEpoch: 1, RoutingRevision: 1,
		RoutingDocumentDigest: "routing-1",
	}
	if reader.advance && reader.locates > 1 {
		location.PublicationID = "publication-2"
		location.RuntimeEpoch = 2
		location.RoutingRevision = 2
		location.RoutingDocumentDigest = "routing-2"
	}
	return location, nil
}

func (reader *publicationRaceReader) LocateCredentialCoherent(
	ctx context.Context,
	kind accesscredential.Kind,
	publicID string,
) (accessruntime.CredentialLocation, error) {
	return reader.LocateCredential(ctx, kind, publicID)
}

func (reader *publicationRaceReader) ReadCredential(
	_ context.Context,
	location accessruntime.CredentialLocation,
	_ accesscredential.Kind,
	_ string,
) (accessprojection.CredentialProjection, error) {
	if location.PublicationID == "publication-1" {
		return accessprojection.CredentialProjection{}, accessruntime.ErrRuntimeCorrupt
	}
	return accessprojection.CredentialProjection{KeyID: reader.projection.KeyID}, nil
}

func (reader *publicationRaceReader) ReadActivePolicy(
	_ context.Context,
	location accessruntime.CredentialLocation,
	keyID string,
) (accessruntime.ActivePolicy, error) {
	return accessruntime.ActivePolicy{
		KeyID: keyID, Revision: reader.projection.Revision, Digest: reader.projection.Digest,
		PublicationID: location.PublicationID, RuntimeEpoch: location.RuntimeEpoch,
		RoutingRevision: location.RoutingRevision, RoutingDocumentDigest: location.RoutingDocumentDigest,
	}, nil
}

func (reader *publicationRaceReader) ReadPolicy(
	context.Context,
	accessruntime.CredentialLocation,
	accessruntime.ActivePolicy,
) (accessprojection.Projection, error) {
	return reader.projection, nil
}

func publicationWaiterProjection(t *testing.T) accessprojection.Projection {
	t.Helper()
	now := time.Date(2026, 8, 25, 1, 2, 3, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "namespace-1", Name: "default", QuotaPartitionID: "partition-1", BillingCurrency: "USD",
		Status: accesscontrol.NamespaceStatusActive, Revision: 1, RuntimeEpoch: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: "user-1", Email: "user@example.com", DisplayName: "User",
		Status: accesscontrol.UserStatusActive, CreatedAt: now, UpdatedAt: now,
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: "key-1", Name: "User key", Owner: user.SubjectRef(),
		Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	projection, err := accessprojection.Compile(accessprojection.Candidate{
		Revision: 2, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &user},
	}, accessprojection.CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return projection
}
