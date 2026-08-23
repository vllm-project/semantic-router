package publicationreplica

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRedisManagersCoordinateTwoReplicasAcrossPublicationKinds(t *testing.T) {
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
	prefix := "publication-replica-it:" + strconv.FormatInt(time.Now().UnixNano(), 10)
	t.Cleanup(func() { deleteIntegrationPrefix(context.Background(), client, prefix+":*") })
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: client, KeyPrefix: prefix, ReplicaLease: 2 * time.Second, MaxNamespaces: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	first := emptyIntegrationPublication(t, 1)
	firstPlan := stageIntegrationPublication(t, ctx, store, first)
	managerA, snapshotsA, stopA := runRedisManager(t, store, "replica-a")
	defer stopA()
	managerB, snapshotsB, stopB := runRedisManager(t, store, "replica-b")
	defer stopB()
	waitFor(t, "two first-publication acknowledgements", func() bool {
		status, statusErr := store.RoutingAcknowledgements(ctx, firstPlan)
		return statusErr == nil && status.Complete() && len(status.Required) == 2
	})
	if managerA.Ready() == nil || managerB.Ready() == nil {
		t.Fatal("a replica became ready before the first publication activated")
	}
	if snapshotsA.wasActivated(first.ID) || snapshotsB.wasActivated(first.ID) {
		t.Fatal("a replica switched to a staged first publication")
	}
	activateIntegrationPublication(t, ctx, store, firstPlan)
	waitForBothCurrent(t, managerA, managerB, first)

	second := integrationPublication(t, 2, false)
	secondPlan := stageIntegrationPublication(t, ctx, store, second)
	if secondPlan.Restrictive() {
		t.Fatal("unchanged routing resources were classified as restrictive")
	}
	waitFor(t, "two nonrestrictive acknowledgements", func() bool {
		status, statusErr := store.RoutingAcknowledgements(ctx, secondPlan)
		return statusErr == nil && status.Complete() && len(status.Required) == 2
	})
	if snapshotsA.wasActivated(second.ID) || snapshotsB.wasActivated(second.ID) {
		t.Fatal("a replica switched to a nonrestrictive candidate before activation")
	}
	activateIntegrationPublication(t, ctx, store, secondPlan)
	waitForBothCurrent(t, managerA, managerB, second)

	third := integrationPublication(t, 3, true)
	thirdPlan := stageIntegrationPublication(t, ctx, store, third)
	if !thirdPlan.Restrictive() {
		t.Fatal("explicit restrictive publication was not classified as restrictive")
	}
	waitFor(t, "two restrictive acknowledgements", func() bool {
		barriers, barrierErr := store.BarrierAcknowledgements(ctx, thirdPlan)
		routing, routingErr := store.RoutingAcknowledgements(ctx, thirdPlan)
		return barrierErr == nil && routingErr == nil && barriers.Complete() && routing.Complete() &&
			len(barriers.Required) == 2 && len(routing.Required) == 2
	})
	if snapshotsA.wasActivated(third.ID) || snapshotsB.wasActivated(third.ID) {
		t.Fatal("a replica switched to a restrictive candidate before activation")
	}
	activateIntegrationPublication(t, ctx, store, thirdPlan)
	waitForBothCurrent(t, managerA, managerB, third)
}

func runRedisManager(
	t *testing.T,
	store *accesspublisher.RedisStore,
	replicaID string,
) (*Manager, *recordingSnapshots, func()) {
	t.Helper()
	snapshots := &recordingSnapshots{activated: make(map[string]bool)}
	manager, err := New(Options{
		Store: store, Snapshots: snapshots, ReplicaID: replicaID,
		DiscoveryInterval: 20 * time.Millisecond, PollInterval: 20 * time.Millisecond, RenewInterval: 100 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- manager.Run(ctx) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("Manager.Run() = %v", err)
				}
			case <-time.After(2 * time.Second):
				t.Error("Manager.Run() did not stop")
			}
		})
	}
	t.Cleanup(stop)
	return manager, snapshots, stop
}

func waitForBothCurrent(t *testing.T, left, right *Manager, publication accesspublisher.Publication) {
	t.Helper()
	waitFor(t, "both replicas to switch active publication", func() bool {
		leftIdentity, leftReady := left.Current(publication.NamespaceID)
		rightIdentity, rightReady := right.Current(publication.NamespaceID)
		return leftReady && rightReady && leftIdentity.PublicationID == publication.ID &&
			rightIdentity.PublicationID == publication.ID
	})
}

func stageIntegrationPublication(
	t *testing.T,
	ctx context.Context,
	store *accesspublisher.RedisStore,
	publication accesspublisher.Publication,
) accesspublisher.PublicationPlan {
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
	return plan
}

func activateIntegrationPublication(
	t *testing.T,
	ctx context.Context,
	store *accesspublisher.RedisStore,
	plan accesspublisher.PublicationPlan,
) {
	t.Helper()
	if err := store.Activate(ctx, plan); err != nil {
		t.Fatal(err)
	}
	for attempts := 0; attempts < 20; attempts++ {
		complete, err := store.Compact(ctx, plan, 100)
		if err != nil {
			t.Fatal(err)
		}
		if complete {
			break
		}
		if attempts == 19 {
			t.Fatal("publication compaction did not finish")
		}
	}
	if err := store.MarkApplied(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, plan); err != nil {
		t.Fatal(err)
	}
}

func integrationPublication(t *testing.T, revision uint64, restrictive bool) accesspublisher.Publication {
	t.Helper()
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "namespace-integration", Name: "Integration", QuotaPartitionID: "partition-integration",
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive,
		Revision: accesscontrol.Revision(revision), RuntimeEpoch: 17, CreatedAt: now, UpdatedAt: now,
	}
	inputPrice, outputPrice := "0.10", "0.20"
	bundle := routingsnapshot.Bundle{
		NamespaceID: string(namespace.ID), Revision: int64(revision), Currency: "USD",
		Models: []routingsnapshot.Model{{
			ID: "model-a", Revision: 1,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "model/a", Execution: routingsnapshot.ModelExecution{RequestTimeout: "30s", StreamTimeout: "60s"},
			Pricing: routingsnapshot.ModelPricing{InputCostPerMillionTokens: &inputPrice, OutputCostPerMillionTokens: &outputPrice},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-a", ProviderID: "private", WireFormat: "openai.chat.v1",
				Origin: "https://model.example/v1", ProviderModelID: "model-a",
				Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-a", Revision: 1, Name: "Recipe A",
			Decisions: []routingsnapshot.Decision{{ID: "decision-a", Name: "Decision A", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
			Document:  json.RawMessage(`{"signals":[],"decisions":[]}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "entrypoint-a", Revision: 1, Name: "Entrypoint A", Aliases: []string{"integration/a"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-a", Name: "Rule A", RecipeID: "recipe-a", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-a": {Models: []routingsnapshot.Assignment{{ModelID: "model-a", ModelRevision: 1, Weight: "1"}}},
				},
			}},
		}},
	}
	state := accesspublisher.DesiredState{
		Namespace: namespace, Revision: revision, RevisionTime: now.Add(time.Duration(revision) * time.Millisecond),
		Routing: bundle,
	}
	if restrictive {
		state.BarrierHints = []accesspublisher.Barrier{{Kind: "model", ResourceID: "model-a", Reason: "test_restriction"}}
	}
	publication, err := accesspublisher.Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	return publication
}

func emptyIntegrationPublication(t *testing.T, revision uint64) accesspublisher.Publication {
	t.Helper()
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "namespace-integration", Name: "Integration", QuotaPartitionID: "partition-integration",
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive,
		Revision: accesscontrol.Revision(revision), RuntimeEpoch: 17, CreatedAt: now, UpdatedAt: now,
	}
	publication, err := accesspublisher.Compile(accesspublisher.DesiredState{
		Namespace: namespace, Revision: revision, RevisionTime: now.Add(time.Duration(revision) * time.Millisecond),
		Routing: routingsnapshot.Bundle{
			NamespaceID: string(namespace.ID), Revision: int64(revision), Currency: namespace.BillingCurrency,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if publication.Routing.ResourceDigests == nil || publication.Manifest.RoutingResources == nil ||
		len(publication.Routing.ResourceDigests) != 0 || len(publication.Manifest.RoutingResources) != 0 {
		t.Fatalf(
			"empty publication is not a canonical empty resource set: routing=%#v manifest=%#v",
			publication.Routing.ResourceDigests,
			publication.Manifest.RoutingResources,
		)
	}
	return publication
}

func deleteIntegrationPrefix(ctx context.Context, client *redis.Client, pattern string) {
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

type recordingSnapshots struct {
	mu        sync.Mutex
	activated map[string]bool
}

func (s *recordingSnapshots) Warm(context.Context, accesspublisher.LoadedRoutingPublication) error {
	return nil
}

func (s *recordingSnapshots) Activate(_ context.Context, publication accesspublisher.LoadedRoutingPublication) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activated[publication.Identity.PublicationID] = true
	return nil
}

func (s *recordingSnapshots) Remove(context.Context, accesspublisher.NamespacePublication) error {
	return nil
}

func (s *recordingSnapshots) wasActivated(publicationID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.activated[publicationID]
}
