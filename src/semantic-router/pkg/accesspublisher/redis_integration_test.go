package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
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

	first := mustPublication(t, 1, "100")
	firstPlan, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := store.Prepare(ctx, first)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil {
		t.Fatal(testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	if err := store.Stage(ctx, firstPlan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, firstPlan); err != nil {
		t.Fatal(err)
	}
	keys, _ := NewKeyspace(prefix, first.NamespaceID, first.QuotaPartition)
	if gate := client.HGet(ctx, keys.AccessGate(), "publication_id").Val(); gate != "" {
		t.Fatalf("staged expansion became visible before activation: %q", gate)
	}
	if err := store.Activate(ctx, firstPlan); err != nil {
		t.Fatal(err)
	}
	compactAll(t, ctx, store, firstPlan, 2)
	if err := store.MarkApplied(ctx, firstPlan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, firstPlan); err != nil {
		t.Fatal(err)
	}
	readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr := store.Readiness(ctx, first.NamespaceID, first.QuotaPartition)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || !readiness.Ready {
		t.Fatalf("first readiness = %+v, %v", readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}

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
	readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr = store.Readiness(ctx, second.NamespaceID, second.QuotaPartition)
	if testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr != nil || !readiness.Ready || readiness.AppliedRevision != 2 {
		t.Fatalf("second readiness = %+v, %v", readiness, testRedisPublicationStagesAcknowledgesAndActivatesAtomicallyErr)
	}
	restrictiveDiagnostics, err := store.Diagnostics(ctx, second.NamespaceID, second.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	if !restrictiveDiagnostics.BarrierAcknowledgementsRequired ||
		!equalReplicaIDs(restrictiveDiagnostics.BarrierAcknowledgements, []string{"router-a"}) ||
		len(restrictiveDiagnostics.MissingBarrierAcks) != 0 {
		t.Fatalf("restrictive publication barrier diagnostics = %+v", restrictiveDiagnostics)
	}
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

func TestRedisFleetRequirementPinsFirstNamespaceUntilEveryReplicaAcknowledges(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	store.requireFleetReplicas = true
	state := validDesiredState(1, "100")
	state.BarrierHints = []Barrier{{Kind: "api_key", ResourceID: "key-publisher", Reason: "first namespace proof"}}
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := store.Prepare(ctx, publication)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Restrictive() {
		t.Fatal("first namespace publication is not restrictive")
	}
	if err := store.InstallBarriers(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.Stage(ctx, plan); !errors.Is(err, ErrAcknowledgements) {
		t.Fatalf("Stage() without fleet replicas = %v", err)
	}
	keys, err := NewKeyspace(prefix, publication.NamespaceID, publication.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	if stored := client.Exists(ctx, keys.Manifest(publication.ID), keys.RoutingSnapshot(publication.DesiredRevision)).Val(); stored != 0 {
		t.Fatalf("Stage() wrote %d immutable documents before fleet membership was available", stored)
	}
	for _, replicaID := range []string{"router-a", "router-b"} {
		if _, err := store.RegisterFleetReplica(ctx, replicaID); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if required := sortedUnique(client.SMembers(ctx, keys.PublicationRequiredReplicas(publication.ID)).Val()); !equalReplicaIDs(required, []string{"router-a", "router-b"}) {
		t.Fatalf("staged required replicas = %v", required)
	}
	if _, err := store.RegisterReplica(ctx, publication.NamespaceID, publication.QuotaPartition, ReplicaRegistration{
		ReplicaID: "router-a", RuntimeEpoch: publication.RuntimeEpoch,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeRouting(
		ctx, publication.NamespaceID, publication.QuotaPartition, "router-a", publication.ID, publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeBarriers(
		ctx, publication.NamespaceID, publication.QuotaPartition, "router-a", publication.ID, publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	status, err := store.RoutingAcknowledgements(ctx, plan)
	if err != nil || status.Complete() || !equalReplicaIDs(status.Missing, []string{"router-b"}) {
		t.Fatalf("single-replica acknowledgement status = %+v, %v", status, err)
	}
	if err := store.Activate(ctx, plan); !errors.Is(err, ErrAcknowledgements) {
		t.Fatalf("Activate() without both fleet acknowledgements = %v", err)
	}
	if _, err := store.RegisterReplica(ctx, publication.NamespaceID, publication.QuotaPartition, ReplicaRegistration{
		ReplicaID: "router-b", RuntimeEpoch: publication.RuntimeEpoch,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeRouting(
		ctx, publication.NamespaceID, publication.QuotaPartition, "router-b", publication.ID, publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	if err := store.AcknowledgeBarriers(
		ctx, publication.NamespaceID, publication.QuotaPartition, "router-b", publication.ID, publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	status, err = store.RoutingAcknowledgements(ctx, plan)
	if err != nil || !status.Complete() || !equalReplicaIDs(status.Required, []string{"router-a", "router-b"}) {
		t.Fatalf("two-replica acknowledgement status = %+v, %v", status, err)
	}
	if err := store.Activate(ctx, plan); err != nil {
		t.Fatal(err)
	}
	diagnostics, err := store.Diagnostics(ctx, publication.NamespaceID, publication.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	if !equalReplicaIDs(diagnostics.ActiveReplicas, []string{"router-a", "router-b"}) ||
		!equalReplicaIDs(diagnostics.RecordedRequiredReplicas, diagnostics.ActiveReplicas) ||
		!equalReplicaIDs(diagnostics.BarrierAcknowledgements, diagnostics.ActiveReplicas) ||
		!equalReplicaIDs(diagnostics.RoutingAcknowledgements, diagnostics.ActiveReplicas) {
		t.Fatalf("activated replica diagnostics = %+v", diagnostics)
	}
}

func TestRedisFleetLeaseExpiryAndRenewal(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	firstExpiry, err := store.RegisterFleetReplica(ctx, "router-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.RegisterFleetReplica(ctx, "router-b"); err != nil {
		t.Fatal(err)
	}
	if replicas, err := store.liveFleetReplicas(ctx); err != nil ||
		!equalReplicaIDs(replicas, []string{"router-a", "router-b"}) {
		t.Fatalf("initial live fleet = %v, %v", replicas, err)
	}
	if err := client.ZAdd(ctx, fleetReplicaIndexKey(prefix), redis.Z{Score: 0, Member: "router-b"}).Err(); err != nil {
		t.Fatal(err)
	}
	if replicas, err := store.liveFleetReplicas(ctx); err != nil || !equalReplicaIDs(replicas, []string{"router-a"}) {
		t.Fatalf("fleet after expiry = %v, %v", replicas, err)
	}
	renewedExpiry, err := store.RegisterFleetReplica(ctx, "router-b")
	if err != nil {
		t.Fatal(err)
	}
	if !renewedExpiry.After(time.Now()) || !firstExpiry.After(time.Now()) {
		t.Fatalf("fleet lease expiries are not in the future: first=%s renewed=%s", firstExpiry, renewedExpiry)
	}
	if replicas, err := store.liveFleetReplicas(ctx); err != nil ||
		!equalReplicaIDs(replicas, []string{"router-a", "router-b"}) {
		t.Fatalf("renewed live fleet = %v, %v", replicas, err)
	}
}

func equalReplicaIDs(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func TestRedisConcurrentPrepareKeepsHighestRevisionAsPublicationHead(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	first := publishWithoutReplicas(t, ctx, store, mustPublication(t, 1, "100"))
	_ = first
	third := mustPublication(t, 3, "300")
	fourth := mustPublication(t, 4, "400")
	type outcome struct {
		publication Publication
		plan        PublicationPlan
		err         error
	}
	results := make(chan outcome, 2)
	var wait sync.WaitGroup
	for _, publication := range []Publication{third, fourth} {
		wait.Add(1)
		go func(candidate Publication) {
			defer wait.Done()
			plan, err := store.Prepare(ctx, candidate)
			results <- outcome{publication: candidate, plan: plan, err: err}
		}(publication)
	}
	wait.Wait()
	close(results)
	var thirdPlan PublicationPlan
	for result := range results {
		if result.publication.ID == fourth.ID && result.err != nil {
			t.Fatalf("latest publication prepare failed: %v", result.err)
		}
		if result.publication.ID == third.ID {
			if result.err != nil && !errors.Is(result.err, ErrSuperseded) {
				t.Fatalf("lower publication prepare failed unexpectedly: %v", result.err)
			}
			thirdPlan = result.plan
		}
	}
	keys, _ := NewKeyspace(prefix, fourth.NamespaceID, fourth.QuotaPartition)
	head := client.HGetAll(ctx, keys.PendingPublication()).Val()
	if head["publication_id"] != fourth.ID || head["revision"] != "4" {
		t.Fatalf("publication head = %+v", head)
	}
	if thirdPlan.Publication.ID != "" {
		if err := store.Activate(ctx, thirdPlan); !errors.Is(err, ErrSuperseded) {
			t.Fatalf("superseded publication activated: %v", err)
		}
	}
}

func TestRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilApplied(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	publishWithoutReplicas(t, ctx, store, mustPublication(t, 1, "100"))

	restrictive := mustPublication(t, 3, "50")
	restrictivePlan, testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr := store.Prepare(ctx, restrictive)
	if testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr != nil {
		t.Fatal(testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr)
	}
	if !restrictivePlan.Restrictive() {
		t.Fatal("lower quota publication was not classified as restrictive")
	}
	if err := store.InstallBarriers(ctx, restrictivePlan); err != nil {
		t.Fatal(err)
	}
	if err := store.Stage(ctx, restrictivePlan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, restrictivePlan); err != nil {
		t.Fatal(err)
	}

	newest := mustPublication(t, 4, "400")
	newestPlan, testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr := store.Prepare(ctx, newest)
	if testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr != nil {
		t.Fatal(testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr)
	}
	if err := store.Stage(ctx, newestPlan); err != nil {
		t.Fatalf("newest full publication could not replace superseded pending pointers: %v", err)
	}
	if err := store.ValidateStaged(ctx, newestPlan); err != nil {
		t.Fatal(err)
	}
	keys, _ := NewKeyspace(prefix, newest.NamespaceID, newest.QuotaPartition)
	for label, pointerKey := range map[string]string{
		"access":     keys.AccessPointer("key-publisher"),
		"logical":    keys.LogicalKey("key-publisher"),
		"credential": keys.CredentialPointer(CredentialKindAPIKey, "publisherkid0001"),
	} {
		pointer := client.HGetAll(ctx, pointerKey).Val()
		if pointer["pending_publication_id"] != newest.ID || pointer["pending_revision"] != "4" {
			t.Fatalf("%s pointer was not replaced by newest revision: %+v", label, pointer)
		}
	}
	directoryKey, testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr := keys.CredentialDirectory(CredentialKindAPIKey, "publisherkid0001")
	if testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr != nil {
		t.Fatal(testRedisNewerStagedPublicationSupersedesPointersAndRetainsDenyBarrierUntilAppliedErr)
	}
	directory := client.HGetAll(ctx, directoryKey).Val()
	if directory["pending_publication_id"] != newest.ID || directory["pending_revision"] != "4" {
		t.Fatalf("credential directory was not replaced by newest revision: %+v", directory)
	}

	denyKey := keys.Deny("api_key", "key-publisher")
	if !client.SIsMember(ctx, denyKey, restrictive.ID).Val() {
		t.Fatal("superseded restriction barrier was released before the newest publication applied")
	}
	if err := store.Activate(ctx, restrictivePlan); !errors.Is(err, ErrSuperseded) {
		t.Fatalf("superseded staged publication activated: %v", err)
	}
	if err := store.Activate(ctx, newestPlan); err != nil {
		t.Fatal(err)
	}
	compactAll(t, ctx, store, newestPlan, 1)
	if err := store.MarkApplied(ctx, newestPlan); err != nil {
		t.Fatal(err)
	}
	if !client.SIsMember(ctx, denyKey, restrictive.ID).Val() {
		t.Fatal("superseded restriction barrier was released before applied cleanup")
	}
	if err := store.ClearAppliedBarriers(ctx, newestPlan); err != nil {
		t.Fatal(err)
	}
	if client.Exists(ctx, denyKey).Val() != 0 {
		t.Fatal("superseded restriction barrier remained after newest publication finalized")
	}
	if state := client.HGet(ctx, keys.Publication(restrictive.ID), "state").Val(); state != "superseded" {
		t.Fatalf("superseded publication state = %q", state)
	}
}

func TestRedisStagedValidationRejectsCorruptedImmutableDocument(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	publication := mustPublication(t, 1, "100")
	plan, err := store.Prepare(ctx, publication)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	keys, _ := NewKeyspace(prefix, publication.NamespaceID, publication.QuotaPartition)
	document := publication.Access[0]
	if err := client.HSet(ctx, keys.AccessDocument(document.KeyID, document.DesiredRevision), "document", `{}`).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); !errors.Is(err, ErrStagedCorrupt) {
		t.Fatalf("ValidateStaged() after immutable document corruption = %v", err)
	}
}

func TestRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocument(t *testing.T) {
	store, client, prefix, ctx := redisIntegrationStore(t)
	publication := mustPublication(t, 1, "100")
	plan, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.Prepare(ctx, publication)
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil {
		t.Fatal(testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	if err := store.Stage(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ValidateStaged(ctx, plan); err != nil {
		t.Fatal(err)
	}
	references, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.ListPublicationNamespaces(ctx)
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil || len(references) != 1 || references[0].NamespaceID != publication.NamespaceID ||
		references[0].QuotaPartition != publication.QuotaPartition {
		t.Fatalf("ListPublicationNamespaces() = %+v, %v", references, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	count, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.CountPublicationNamespaces(ctx)
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil || count != 1 {
		t.Fatalf("CountPublicationNamespaces() = %d, %v", count, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	reference, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.GetPublicationNamespace(ctx, publication.NamespaceID)
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil || reference != references[0] {
		t.Fatalf("GetPublicationNamespace() = %+v, %v", reference, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	if _, err := store.GetPublicationNamespace(ctx, "missing-namespace"); !errors.Is(err, ErrNamespaceNotFound) {
		t.Fatalf("GetPublicationNamespace(missing) = %v, want ErrNamespaceNotFound", err)
	}
	heads, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.ReadPublicationHeads(ctx, references[0])
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil || heads.Active != nil || heads.Candidate == nil || heads.Candidate.PublicationID != publication.ID {
		t.Fatalf("ReadPublicationHeads() = %+v, %v", heads, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	loaded, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr := store.LoadRoutingPublication(ctx, *heads.Candidate)
	if testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr != nil || loaded.Identity.PublicationID != publication.ID || loaded.Snapshot.Digest != publication.Routing.Snapshot.Digest {
		t.Fatalf("LoadRoutingPublication() = %+v, %v", loaded.Identity, testRedisReplicaLoaderDiscoversAndRejectsCorruptedRoutingDocumentErr)
	}
	keys, _ := NewKeyspace(prefix, publication.NamespaceID, publication.QuotaPartition)
	if err := client.Set(ctx, keys.RoutingSnapshot(publication.DesiredRevision), `{}`, 0).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadRoutingPublication(ctx, *heads.Candidate); !errors.Is(err, ErrStagedCorrupt) {
		t.Fatalf("LoadRoutingPublication() after corruption = %v", err)
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
