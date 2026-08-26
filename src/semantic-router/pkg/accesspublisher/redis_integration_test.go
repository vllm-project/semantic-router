package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

func TestRedisPublishedProviderCredentialRotationRetentionAndIsolation(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	firstState, codec := desiredStateWithProviderCredential(t, 1)
	firstPublication, err := Compile(firstState)
	if err != nil {
		t.Fatal(err)
	}
	publishWithoutReplicas(t, ctx, store, firstPublication)
	firstIdentity := backendinvoker.CredentialPublication{
		NamespaceID: firstPublication.NamespaceID, QuotaPartition: firstPublication.QuotaPartition,
		PublicationID: firstPublication.ID,
	}
	_, active, err := store.LoadActivePublishedProviderCredential(ctx, firstIdentity, providerFixtureCredentialID)
	if err != nil || active.ID != providerFixtureVersionID {
		t.Fatalf("first active provider credential = %+v, %v", active, err)
	}

	secondState := firstState
	secondState.Revision = 2
	secondState.RevisionTime = firstState.RevisionTime.Add(time.Second)
	secondState.Namespace.Revision = 2
	secondState.Namespace.UpdatedAt = secondState.RevisionTime
	secondState.Routing.Revision = 2
	credential := firstState.ProviderCredentials[0].Credential
	newVersionID := "44444444-4444-4444-8444-444444444444"
	credential.ActiveVersionID = &newVersionID
	credential.Revision = 2
	credential.UpdatedAt = secondState.RevisionTime
	newVersion, err := codec.Seal(credential, newVersionID, []byte("rotated-provider-secret"), secondState.RevisionTime)
	if err != nil {
		t.Fatal(err)
	}
	oldVersion := firstState.ProviderCredentials[0].Versions[0]
	retireAt := secondState.RevisionTime.Add(10 * time.Minute)
	oldVersion.Status = providercredential.VersionRetiring
	oldVersion.ExpiresAt = &retireAt
	secondState.ProviderCredentials = []ProviderCredentialCandidate{{
		Credential: credential, Versions: []providercredential.Version{oldVersion, newVersion},
	}}
	secondPublication, err := Compile(secondState)
	if err != nil {
		t.Fatal(err)
	}
	publishWithoutReplicas(t, ctx, store, secondPublication)
	secondIdentity := backendinvoker.CredentialPublication{
		NamespaceID: secondPublication.NamespaceID, QuotaPartition: secondPublication.QuotaPartition,
		PublicationID: secondPublication.ID,
	}
	_, active, err = store.LoadActivePublishedProviderCredential(ctx, secondIdentity, providerFixtureCredentialID)
	if err != nil || active.ID != newVersionID {
		t.Fatalf("rotated active provider credential = %+v, %v", active, err)
	}
	_, pinned, err := store.LoadPinnedPublishedProviderCredential(
		ctx, secondIdentity, providerFixtureCredentialID, providerFixtureVersionID,
	)
	if err != nil || pinned.Status != providercredential.VersionRetiring {
		t.Fatalf("rotated publication retiring version = %+v, %v", pinned, err)
	}
	// The first immutable publication remains resolvable for dispatches that
	// pinned it before rotation.
	_, pinned, err = store.LoadPinnedPublishedProviderCredential(
		ctx, firstIdentity, providerFixtureCredentialID, providerFixtureVersionID,
	)
	if err != nil || pinned.Status != providercredential.VersionActive {
		t.Fatalf("retained publication pinned version = %+v, %v", pinned, err)
	}

	wrongNamespace := secondIdentity
	wrongNamespace.NamespaceID = "55555555-5555-4555-8555-555555555555"
	if _, _, err := store.LoadActivePublishedProviderCredential(
		ctx, wrongNamespace, providerFixtureCredentialID,
	); !errors.Is(err, providercredential.ErrUnavailable) {
		t.Fatalf("cross-namespace provider credential load = %v", err)
	}
	wrongPartition := secondIdentity
	wrongPartition.QuotaPartition = "partition-other"
	if _, _, err := store.LoadActivePublishedProviderCredential(
		ctx, wrongPartition, providerFixtureCredentialID,
	); !errors.Is(err, providercredential.ErrUnavailable) {
		t.Fatalf("cross-partition provider credential load = %v", err)
	}
	wrongPublication := secondIdentity
	wrongPublication.PublicationID = "pub_missing"
	if _, _, err := store.LoadActivePublishedProviderCredential(
		ctx, wrongPublication, providerFixtureCredentialID,
	); !errors.Is(err, providercredential.ErrUnavailable) {
		t.Fatalf("cross-publication provider credential load = %v", err)
	}

	keys, _ := NewKeyspace(prefix, secondPublication.NamespaceID, secondPublication.QuotaPartition)
	documentKey := keys.ProviderCredentialDocument(secondPublication.ID, providerFixtureCredentialID)
	if err := client.Set(ctx, documentKey, `{}`, 0).Err(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.LoadActivePublishedProviderCredential(
		ctx, secondIdentity, providerFixtureCredentialID,
	); !errors.Is(err, ErrStagedCorrupt) {
		t.Fatalf("tampered published provider credential load = %v", err)
	}
}

func TestRedisInactiveProviderCredentialFailsClosedWithoutErasingPriorPublication(t *testing.T) {
	for _, status := range []providercredential.Status{
		providercredential.StatusDisabled,
		providercredential.StatusDeleted,
	} {
		t.Run(string(status), func(t *testing.T) {
			store, _, _, ctx := redisIntegrationStore(t)
			activeState, _ := desiredStateWithProviderCredential(t, 1)
			activePublication, err := Compile(activeState)
			if err != nil {
				t.Fatal(err)
			}
			publishWithoutReplicas(t, ctx, store, activePublication)

			inactiveState := activeState
			inactiveState.Revision = 2
			inactiveState.RevisionTime = activeState.RevisionTime.Add(time.Second)
			inactiveState.Namespace.Revision = 2
			inactiveState.Namespace.UpdatedAt = inactiveState.RevisionTime
			inactiveState.Routing.Revision = 2
			inactiveCredential := activeState.ProviderCredentials[0].Credential
			inactiveCredential.Status = status
			inactiveCredential.ActiveVersionID = nil
			inactiveCredential.Revision = 2
			inactiveCredential.UpdatedAt = inactiveState.RevisionTime
			if status == providercredential.StatusDeleted {
				deletedAt := inactiveCredential.UpdatedAt
				inactiveCredential.DeletedAt = &deletedAt
			}
			inactiveState.ProviderCredentials = []ProviderCredentialCandidate{{Credential: inactiveCredential}}
			inactivePublication, err := Compile(inactiveState)
			if err != nil {
				t.Fatal(err)
			}
			plan := publishWithoutReplicas(t, ctx, store, inactivePublication)
			if !plan.Restrictive() {
				t.Fatalf("provider credential %s did not create a restrictive publication", status)
			}
			inactiveIdentity := backendinvoker.CredentialPublication{
				NamespaceID: inactivePublication.NamespaceID, QuotaPartition: inactivePublication.QuotaPartition,
				PublicationID: inactivePublication.ID,
			}
			if _, _, err := store.LoadActivePublishedProviderCredential(
				ctx, inactiveIdentity, providerFixtureCredentialID,
			); !errors.Is(err, providercredential.ErrUnavailable) {
				t.Fatalf("%s provider credential active load = %v", status, err)
			}
			activeIdentity := backendinvoker.CredentialPublication{
				NamespaceID: activePublication.NamespaceID, QuotaPartition: activePublication.QuotaPartition,
				PublicationID: activePublication.ID,
			}
			if _, _, err := store.LoadPinnedPublishedProviderCredential(
				ctx, activeIdentity, providerFixtureCredentialID, providerFixtureVersionID,
			); err != nil {
				t.Fatalf("prior publication was erased by %s: %v", status, err)
			}
		})
	}
}

func TestRedisPublicationStagesAcknowledgesAndActivatesAtomically(t *testing.T) {
	address := os.Getenv("ACCESSPUBLISHER_REDIS_ADDR")
	if address == "" {
		t.Skip("ACCESSPUBLISHER_REDIS_ADDR is not configured")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := fmt.Sprintf("access-publisher-it:%d", time.Now().UnixNano())
	t.Cleanup(func() { deleteRedisPrefix(context.Background(), client, prefix+":*") })
	store, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := NewRedisStore(RedisStoreOptions{Client: client, KeyPrefix: prefix, ReplicaLease: 20 * time.Second})
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil {
		t.Fatal(testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}

	first, keys := publishInitialRedisPlan(t, ctx, store, client, prefix)

	if _, err := store.RegisterReplica(ctx, first.NamespaceID, first.QuotaPartition, ReplicaRegistration{
		ReplicaID: "router-a", RuntimeEpoch: first.RuntimeEpoch,
		AccessPublication: first.ID, RoutingPublication: first.ID,
	}); err != nil {
		t.Fatal(err)
	}
	expansiveDiagnostics, err := store.Diagnostics(ctx, first.NamespaceID, first.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	if expansiveDiagnostics.BarrierAcknowledgementsRequired || len(expansiveDiagnostics.BarrierAcknowledgements) != 0 ||
		len(expansiveDiagnostics.MissingBarrierAcks) != 0 {
		t.Fatalf("expansive publication barrier diagnostics = %+v", expansiveDiagnostics)
	}
	second := mustPublication(t, 2, "50")
	secondPlan, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := store.Prepare(ctx, second)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil {
		t.Fatal(testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	if !secondPlan.Restrictive() {
		t.Fatal("quota reduction was not classified as restrictive")
	}
	if err := store.InstallBarriers(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	if err := store.Stage(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	denyKey := keys.Deny("api_key", "key-publisher")
	if member := client.SIsMember(ctx, denyKey, second.ID).Val(); !member {
		t.Fatal("restriction barrier was not installed before staging completed")
	}
	pointer := client.HGetAll(ctx, keys.AccessPointer("key-publisher")).Val()
	selected, _, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := SelectPointer(pointer, first.ID)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || selected["revision"] != "1" {
		t.Fatalf("old gate did not keep old pointer active: %+v, %v", selected, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	barrierStatus, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := store.BarrierAcknowledgements(ctx, secondPlan)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || len(barrierStatus.Missing) != 1 || barrierStatus.Missing[0] != "router-a" {
		t.Fatalf("barrier acknowledgements = %+v, %v", barrierStatus, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	if err := store.Activate(ctx, secondPlan); !errors.Is(err, ErrAcknowledgements) {
		t.Fatalf("Activate() without acknowledgements = %v", err)
	}
	if err := store.AcknowledgeBarriers(ctx, second.NamespaceID, second.QuotaPartition, "router-a", second.ID, second.Digest); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeRouting(ctx, second.NamespaceID, second.QuotaPartition, "router-a", second.ID, second.Digest); err != nil {
		t.Fatal(err)
	}
	if err := store.Activate(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	pointer = client.HGetAll(ctx, keys.AccessPointer("key-publisher")).Val()
	selected, _, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr = SelectPointer(pointer, second.ID)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || selected["revision"] != "2" {
		t.Fatalf("new gate did not select pending pointer: %+v, %v", selected, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	compactAll(t, ctx, store, secondPlan, 1)
	if err := store.MarkApplied(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, secondPlan); err != nil {
		t.Fatal(err)
	}
	if client.Exists(ctx, denyKey).Val() != 0 {
		t.Fatal("restriction barrier remained after the applied watermark")
	}
	readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := store.Readiness(ctx, second.NamespaceID, second.QuotaPartition)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || !readiness.Ready ||
		readiness.AppliedRevision != 2 || readiness.RoutingDigest != second.Routing.Digest {
		t.Fatalf("second readiness = %+v, %v", readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	assertRestrictiveDiagnostics(t, ctx, store, second)
}

func assertRestrictiveDiagnostics(t *testing.T, ctx context.Context, store *RedisStore, publication Publication) {
	t.Helper()
	diagnostics, err := store.Diagnostics(ctx, publication.NamespaceID, publication.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	if !diagnostics.BarrierAcknowledgementsRequired ||
		!equalReplicaIDs(diagnostics.BarrierAcknowledgements, []string{"router-a"}) ||
		len(diagnostics.MissingBarrierAcks) != 0 {
		t.Fatalf("restrictive publication barrier diagnostics = %+v", diagnostics)
	}
}

func publishInitialRedisPlan(
	t *testing.T,
	ctx context.Context,
	store *RedisStore,
	client *redis.Client,
	prefix string,
) (Publication, Keyspace) {
	t.Helper()
	publication := mustPublication(t, 1, "100")
	plan, prepareErr := store.Prepare(ctx, publication)
	if prepareErr != nil {
		t.Fatal(prepareErr)
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		t.Fatal(err)
	}
	keys, _ := NewKeyspace(prefix, publication.NamespaceID, publication.QuotaPartition)
	if gate := client.HGet(ctx, keys.AccessGate(), "publication_id").Val(); gate != "" {
		t.Fatalf("staged expansion became visible before activation: %q", gate)
	}
	if err := store.Activate(ctx, plan); err != nil {
		t.Fatal(err)
	}
	compactAll(t, ctx, store, plan, 2)
	if err := store.MarkApplied(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, plan); err != nil {
		t.Fatal(err)
	}
	readiness, readinessErr := store.Readiness(ctx, publication.NamespaceID, publication.QuotaPartition)
	if readinessErr != nil || !readiness.Ready {
		t.Fatalf("first readiness = %+v, %v", readiness, readinessErr)
	}
	return publication, keys
}

func TestRedisActivationClosesReplicaJoinRaceAndExpiresLeases(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	first := publishWithoutReplicas(t, ctx, store, mustPublication(t, 1, "100"))
	for _, replica := range []string{"router-a", "router-b"} {
		if _, err := store.RegisterReplica(ctx, first.Publication.NamespaceID, first.Publication.QuotaPartition, ReplicaRegistration{
			ReplicaID: replica, RuntimeEpoch: first.Publication.RuntimeEpoch,
			AccessPublication: first.Publication.ID, RoutingPublication: first.Publication.ID,
		}); err != nil {
			t.Fatal(err)
		}
	}
	second := mustPublication(t, 2, "200")
	plan, testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr := store.Prepare(ctx, second)
	if testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr != nil {
		t.Fatal(testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr)
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeRouting(ctx, second.NamespaceID, second.QuotaPartition, "router-a", second.ID, second.Digest); err != nil {
		t.Fatal(err)
	}
	if err := store.Activate(ctx, plan); !errors.Is(err, ErrAcknowledgements) {
		t.Fatalf("newly joined unacknowledged replica did not block activation: %v", err)
	}
	keys, _ := NewKeyspace(prefix, second.NamespaceID, second.QuotaPartition)
	if err := client.ZAdd(ctx, keys.ReplicaIndex(), redis.Z{Score: 0, Member: "router-b"}).Err(); err != nil {
		t.Fatal(err)
	}
	status, testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr := store.RoutingAcknowledgements(ctx, plan)
	if testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr != nil || len(status.Required) != 1 || status.Required[0] != "router-a" || !status.Complete() {
		t.Fatalf("expired lease acknowledgement status = %+v, %v", status, testRedisActivationClosesReplicaJoinRaceAndExpiresLeasesErr)
	}
	if err := store.Activate(ctx, plan); err != nil {
		t.Fatal(err)
	}
}

func TestRedisActiveReplicaReadinessFailsClosedWithoutADataPlane(t *testing.T) {
	store, _, _, ctx := redisIntegrationStore(t)
	publication := mustPublication(t, 1, "100")
	publishWithoutReplicas(t, ctx, store, publication)

	status, err := store.ActiveReplicaAcknowledgements(
		ctx, publication.NamespaceID, publication.QuotaPartition, ActiveGeneration{
			PublicationID: publication.ID, Revision: publication.DesiredRevision,
			RuntimeEpoch: publication.RuntimeEpoch, RoutingSnapshotDigest: publication.Routing.Digest,
		},
	)
	if !errors.Is(err, ErrAcknowledgements) || status.Complete() {
		t.Fatalf("empty active replica readiness = %+v, %v", status, err)
	}
}

func redisIntegrationStore(t *testing.T) (*RedisStore, *redis.Client, string, context.Context) {
	t.Helper()
	address := os.Getenv("ACCESSPUBLISHER_REDIS_ADDR")
	if address == "" {
		t.Skip("ACCESSPUBLISHER_REDIS_ADDR is not configured")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := "access-publisher-it:" + strconv.FormatInt(time.Now().UnixNano(), 10)
	t.Cleanup(func() { deleteRedisPrefix(context.Background(), client, prefix+":*") })
	store, err := NewRedisStore(RedisStoreOptions{Client: client, KeyPrefix: prefix, ReplicaLease: 20 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	return store, client, prefix, ctx
}

func publishWithoutReplicas(t *testing.T, ctx context.Context, store *RedisStore, publication Publication) PublicationPlan {
	t.Helper()
	plan, err := store.Prepare(ctx, publication)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Restrictive() {
		if err := store.InstallBarriers(ctx, plan); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.Activate(ctx, plan); err != nil {
		t.Fatal(err)
	}
	compactAll(t, ctx, store, plan, 2)
	if err := store.MarkApplied(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, plan); err != nil {
		t.Fatal(err)
	}
	return plan
}

func compactAll(t *testing.T, ctx context.Context, store *RedisStore, plan PublicationPlan, batch int) {
	t.Helper()
	for attempts := 0; attempts < 100; attempts++ {
		complete, err := store.Compact(ctx, plan, batch)
		if err != nil {
			t.Fatal(err)
		}
		if complete {
			return
		}
	}
	t.Fatal("publication compaction did not complete")
}

func deleteRedisPrefix(ctx context.Context, client *redis.Client, pattern string) {
	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return
		}
		if len(keys) > 0 {
			_ = client.Del(ctx, keys...).Err()
		}
		cursor = next
		if cursor == 0 {
			return
		}
	}
}
