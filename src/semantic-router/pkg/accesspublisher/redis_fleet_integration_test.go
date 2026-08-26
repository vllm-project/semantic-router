package accesspublisher

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

func TestRedisFleetRequirementPinsFirstNamespaceUntilEveryReplicaAcknowledges(t *testing.T) {
	fixture := prepareRedisFleetRequirement(t)
	assertRedisFleetAcknowledgements(t, fixture)
	assertRedisFleetActivationReadiness(t, fixture)
	assertRedisFleetRejoinAndDiagnostics(t, fixture)
}

type redisFleetRequirementFixture struct {
	store       *RedisStore
	client      *redis.Client
	ctx         context.Context
	publication Publication
	plan        PublicationPlan
	keys        Keyspace
}

func prepareRedisFleetRequirement(t *testing.T) redisFleetRequirementFixture {
	t.Helper()
	store, client, prefix, ctx := redisIntegrationStore(t)
	store.requireFleetReplicas = true
	state := validDesiredState(1, "100")
	state.BarrierHints = []Barrier{{Kind: "api_key", ResourceID: "key-publisher", Reason: "first namespace proof"}}
	publication, compileErr := Compile(state)
	if compileErr != nil {
		t.Fatal(compileErr)
	}
	plan, prepareErr := store.Prepare(ctx, publication)
	if prepareErr != nil {
		t.Fatal(prepareErr)
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
	keys, keyspaceErr := NewKeyspace(prefix, publication.NamespaceID, publication.QuotaPartition)
	if keyspaceErr != nil {
		t.Fatal(keyspaceErr)
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
	return redisFleetRequirementFixture{
		store: store, client: client, ctx: ctx, publication: publication, plan: plan, keys: keys,
	}
}

func assertRedisFleetAcknowledgements(t *testing.T, fixture redisFleetRequirementFixture) {
	t.Helper()
	status, statusErr := fixture.store.RoutingAcknowledgements(fixture.ctx, fixture.plan)
	if statusErr != nil || status.Complete() || !equalReplicaIDs(status.Missing, []string{"router-b"}) {
		t.Fatalf("single-replica acknowledgement status = %+v, %v", status, statusErr)
	}
	if err := fixture.store.Activate(fixture.ctx, fixture.plan); !errors.Is(err, ErrAcknowledgements) {
		t.Fatalf("Activate() without both fleet acknowledgements = %v", err)
	}
	if _, err := fixture.store.RegisterReplica(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
		ReplicaRegistration{ReplicaID: "router-b", RuntimeEpoch: fixture.publication.RuntimeEpoch},
	); err != nil {
		t.Fatal(err)
	}
	if err := fixture.store.AcknowledgeRouting(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
		"router-b", fixture.publication.ID, fixture.publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	if err := fixture.store.AcknowledgeBarriers(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
		"router-b", fixture.publication.ID, fixture.publication.Digest,
	); err != nil {
		t.Fatal(err)
	}
	status, statusErr = fixture.store.RoutingAcknowledgements(fixture.ctx, fixture.plan)
	if statusErr != nil || !status.Complete() || !equalReplicaIDs(status.Required, []string{"router-a", "router-b"}) {
		t.Fatalf("two-replica acknowledgement status = %+v, %v", status, statusErr)
	}
	if err := fixture.store.Activate(fixture.ctx, fixture.plan); err != nil {
		t.Fatal(err)
	}
}

func assertRedisFleetActivationReadiness(t *testing.T, fixture redisFleetRequirementFixture) {
	t.Helper()
	expected := ActiveGeneration{
		PublicationID: fixture.publication.ID, Revision: fixture.publication.DesiredRevision,
		RuntimeEpoch: fixture.publication.RuntimeEpoch, RoutingSnapshotDigest: fixture.publication.Routing.Digest,
	}
	activated, statusErr := fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, expected,
	)
	if statusErr != nil || activated.Complete() ||
		!equalReplicaIDs(activated.Missing, []string{"router-a", "router-b"}) {
		t.Fatalf("candidate-only active replica status = %+v, %v", activated, statusErr)
	}
	if _, err := fixture.store.RegisterReplica(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
		ReplicaRegistration{
			ReplicaID: "router-a", RuntimeEpoch: fixture.publication.RuntimeEpoch,
			AccessPublication: fixture.publication.ID, RoutingPublication: fixture.publication.ID,
		},
	); err != nil {
		t.Fatal(err)
	}
	activated, statusErr = fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, expected,
	)
	if statusErr != nil || activated.Complete() || !equalReplicaIDs(activated.Missing, []string{"router-b"}) {
		t.Fatalf("single activated replica status = %+v, %v", activated, statusErr)
	}
	// A candidate participant that exits before installing the active
	// generation must stop blocking delivery once its namespace lease expires.
	if err := fixture.client.ZAdd(
		fixture.ctx, fixture.keys.ReplicaIndex(), redis.Z{Score: 0, Member: "router-b"},
	).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, expected,
	); !errors.Is(err, ErrPublicationChanged) {
		t.Fatalf("expired membership transition = %v, want publication changed retry", err)
	}
	activated, statusErr = fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, expected,
	)
	if statusErr != nil || !activated.Complete() || !equalReplicaIDs(activated.Required, []string{"router-a"}) {
		t.Fatalf("readiness after expired replica retirement = %+v, %v", activated, statusErr)
	}
}

func assertRedisFleetRejoinAndDiagnostics(t *testing.T, fixture redisFleetRequirementFixture) {
	t.Helper()
	if _, err := fixture.store.RegisterReplica(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
		ReplicaRegistration{
			ReplicaID: "router-b", RuntimeEpoch: fixture.publication.RuntimeEpoch,
			AccessPublication: fixture.publication.ID, RoutingPublication: fixture.publication.ID,
		},
	); err != nil {
		t.Fatal(err)
	}
	expected := ActiveGeneration{
		PublicationID: fixture.publication.ID, Revision: fixture.publication.DesiredRevision,
		RuntimeEpoch: fixture.publication.RuntimeEpoch, RoutingSnapshotDigest: fixture.publication.Routing.Digest,
	}
	activated, statusErr := fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, expected,
	)
	if statusErr != nil || !activated.Complete() ||
		!equalReplicaIDs(activated.Required, []string{"router-a", "router-b"}) {
		t.Fatalf("all activated replica status = %+v, %v", activated, statusErr)
	}
	stale := expected
	stale.PublicationID += "-stale"
	if _, err := fixture.store.ActiveReplicaAcknowledgements(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition, stale,
	); !errors.Is(err, ErrPublicationChanged) {
		t.Fatalf("stale expected generation = %v, want publication changed", err)
	}
	diagnostics, diagnosticsErr := fixture.store.Diagnostics(
		fixture.ctx, fixture.publication.NamespaceID, fixture.publication.QuotaPartition,
	)
	if diagnosticsErr != nil {
		t.Fatal(diagnosticsErr)
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
	firstExpiry, registrationErr := store.RegisterFleetReplica(ctx, "router-a")
	if registrationErr != nil {
		t.Fatal(registrationErr)
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
	renewedExpiry, renewalErr := store.RegisterFleetReplica(ctx, "router-b")
	if renewalErr != nil {
		t.Fatal(renewalErr)
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
