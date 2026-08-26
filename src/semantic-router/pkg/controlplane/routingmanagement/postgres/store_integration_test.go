package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const routingCatalogRevision = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func newRoutingStore(t *testing.T, db *sql.DB) *Store {
	t.Helper()
	store, err := New(db, config.ValidateDurableRoutingSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func TestRoutingListsApplyExactScopeBeforeStablePagination(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	models := make([]routingsnapshot.Model, 4)
	for index, id := range []string{"model_one", "model_two", "model_three", "model_four"} {
		models[index] = routingTestModel(1, id)
		models[index].ID, models[index].Name = id, id
		models[index].Backends[0].ID = uuid.NewString()
	}
	seedMeta := mutationMeta("seed scoped Models")
	if _, _, err := store.CreateModels(ctx, namespaceID, models, seedMeta); err != nil {
		t.Fatal(err)
	}
	var emptyActorChain bool
	if err := db.QueryRowContext(ctx, `SELECT actor_chain = '[]'::jsonb
FROM access_audit_events
WHERE namespace_id=$1 AND request_id=$2`, namespaceID, seedMeta.RequestID).Scan(&emptyActorChain); err != nil {
		t.Fatalf("read routing audit actor chain: %v", err)
	}
	if !emptyActorChain {
		t.Fatal("routing audit actor chain was not persisted as an empty array")
	}
	modelScope := exactRoutingScope(namespaceID, accesscontrol.ScopeResourceModel,
		"model_one", "model_three", "model_four")
	first, testRoutingListsApplyExactScopeBeforeStablePaginationErr := store.ListModels(ctx, namespaceID, routingmanagement.ListQuery{Limit: 2, Scope: modelScope})
	if testRoutingListsApplyExactScopeBeforeStablePaginationErr != nil || !first.HasMore || len(first.Items) != 2 {
		t.Fatalf("first scoped Model page = %#v, %v", first, testRoutingListsApplyExactScopeBeforeStablePaginationErr)
	}
	last := first.Items[len(first.Items)-1]
	second, testRoutingListsApplyExactScopeBeforeStablePaginationErr := store.ListModels(ctx, namespaceID, routingmanagement.ListQuery{
		Limit: 2, Scope: modelScope,
		After: &routingmanagement.ListCursor{CreatedAt: last.CreatedAt, ID: last.ID},
	})
	if testRoutingListsApplyExactScopeBeforeStablePaginationErr != nil || second.HasMore || len(second.Items) != 1 {
		t.Fatalf("second scoped Model page = %#v, %v", second, testRoutingListsApplyExactScopeBeforeStablePaginationErr)
	}
	visible := map[string]bool{}
	for _, page := range [][]routingmanagement.Model{first.Items, second.Items} {
		for _, model := range page {
			if visible[model.ID] || model.ID == "model_two" {
				t.Fatalf("unstable or unauthorized Model page: %q", model.ID)
			}
			visible[model.ID] = true
		}
	}
	if len(visible) != 3 {
		t.Fatalf("visible Models = %#v", visible)
	}
	firstRecipe := routingTestRecipe(1, "Simple")
	secondRecipe := routingTestRecipe(1, "Simple two")
	secondRecipe.ID, secondRecipe.Name = "recipe_two", "Recipe Two"
	if _, _, err := store.CreateRecipe(ctx, namespaceID, "", firstRecipe, mutationMeta("seed Recipe one")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.CreateRecipe(ctx, namespaceID, "", secondRecipe, mutationMeta("seed Recipe two")); err != nil {
		t.Fatal(err)
	}
	recipePage, testRoutingListsApplyExactScopeBeforeStablePaginationErr := store.ListRecipes(ctx, namespaceID, routingmanagement.ListQuery{
		Limit: 50, Scope: exactRoutingScope(namespaceID, accesscontrol.ScopeResourceRecipe, "recipe_two"),
	})
	if testRoutingListsApplyExactScopeBeforeStablePaginationErr != nil || len(recipePage.Items) != 1 || recipePage.Items[0].ID != "recipe_two" {
		t.Fatalf("exact Recipe page = %#v, %v", recipePage, testRoutingListsApplyExactScopeBeforeStablePaginationErr)
	}

	firstEntrypoint := routingTestEntrypoint(1, 1, 1)
	secondEntrypoint := routingTestEntrypoint(1, 1, 1)
	secondEntrypoint.ID, secondEntrypoint.Name = "entrypoint_two", "Entrypoint Two"
	secondEntrypoint.Aliases = []string{"vllm-sr/two"}
	secondEntrypoint.Rules[0].ID = "rule_two"
	secondEntrypoint.Rules[0].RecipeID = "recipe_two"
	secondEntrypoint.Rules[0].Assignments["decision_simple"] = routingsnapshot.AssignmentSet{
		Models: []routingsnapshot.Assignment{{ModelID: "model_two", ModelRevision: 1, Weight: "1"}},
	}
	if _, _, err := store.CreateEntrypoint(ctx, namespaceID, firstEntrypoint, mutationMeta("seed Entrypoint one")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.CreateEntrypoint(ctx, namespaceID, secondEntrypoint, mutationMeta("seed Entrypoint two")); err != nil {
		t.Fatal(err)
	}
	entrypointPage, testRoutingListsApplyExactScopeBeforeStablePaginationErr := store.ListEntrypoints(ctx, namespaceID, routingmanagement.ListQuery{
		Limit: 50, Scope: exactRoutingScope(namespaceID, accesscontrol.ScopeResourceEntrypoint, "entrypoint_two"),
	})
	if testRoutingListsApplyExactScopeBeforeStablePaginationErr != nil || len(entrypointPage.Items) != 1 || entrypointPage.Items[0].ID != "entrypoint_two" ||
		entrypointPage.Items[0].RuleCount != 1 || entrypointPage.Items[0].AssignedModelCount != 1 ||
		len(entrypointPage.Items[0].RecipeIDs) != 1 || entrypointPage.Items[0].RecipeIDs[0] != "recipe_two" ||
		len(entrypointPage.Items[0].Current.Rules) != 0 {
		t.Fatalf("exact Entrypoint page = %#v, %v", entrypointPage, testRoutingListsApplyExactScopeBeforeStablePaginationErr)
	}
}

func exactRoutingScope(
	namespaceID string,
	resourceType accesscontrol.ScopeResourceType,
	ids ...string,
) accesscontrol.ResultScope {
	resources := make([]accesscontrol.ResourceID, len(ids))
	for index, id := range ids {
		resources[index] = accesscontrol.ResourceID(id)
	}
	return accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{resourceType: resources},
	}
}

func TestPublishedClosurePinsExactImmutableRevisions(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	meta := mutationMeta("initial authoring")

	modelV1 := routingTestModel(1, "provider-model-v1")
	if _, receipt, err := store.CreateModels(ctx, namespaceID, []routingsnapshot.Model{modelV1}, meta); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("CreateModels() = %+v, %v", receipt, err)
	}
	recipeV1 := routingTestRecipe(1, "Simple")
	if _, receipt, err := store.CreateRecipe(ctx, namespaceID, "", recipeV1, meta); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("CreateRecipe() = %+v, %v", receipt, err)
	}
	entrypointV1 := routingTestEntrypoint(1, 1, 1)
	if _, receipt, err := store.CreateEntrypoint(ctx, namespaceID, entrypointV1, meta); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("CreateEntrypoint() = %+v, %v", receipt, err)
	}
	assertOutboxCount(t, ctx, db, 0)

	firstSnapshot, firstReceipt, publishEntrypointErr := store.PublishEntrypoint(
		ctx, namespaceID, entrypointV1.ID, 1, mutationMeta("publish first closure"),
	)
	if publishEntrypointErr != nil || firstReceipt.DesiredRevision != 1 {
		t.Fatalf("first PublishEntrypoint() = %+v, %v", firstReceipt, publishEntrypointErr)
	}
	assertClosureRevisions(t, firstSnapshot, 1, 1, 1)
	firstPinned := loadPublishedSnapshot(t, ctx, db, namespaceID, firstReceipt.DesiredRevision)
	assertOutboxCount(t, ctx, db, 1)

	modelV2 := routingTestModel(2, "provider-model-v2")
	if _, receipt, err := store.UpdateModel(ctx, namespaceID, modelV1.ID, 1, modelV2, mutationMeta("edit Model draft")); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("UpdateModel() = %+v, %v", receipt, err)
	}
	recipeV2 := routingTestRecipe(2, "Simple v2")
	if _, receipt, err := store.UpdateRecipe(ctx, namespaceID, recipeV1.ID, 1, "", recipeV2, mutationMeta("edit Recipe draft")); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("UpdateRecipe() = %+v, %v", receipt, err)
	}
	entrypointV2 := routingTestEntrypoint(2, 2, 2)
	if _, receipt, err := store.UpdateEntrypoint(ctx, namespaceID, entrypointV1.ID, 2, entrypointV2, mutationMeta("edit Entrypoint draft")); err != nil || receipt.DesiredRevision != 0 {
		t.Fatalf("UpdateEntrypoint() = %+v, %v", receipt, err)
	}
	assertOutboxCount(t, ctx, db, 1)
	afterDraft := loadPublishedSnapshot(t, ctx, db, namespaceID, firstReceipt.DesiredRevision)
	if afterDraft.Digest != firstPinned.Digest {
		t.Fatalf("draft edit changed active closure digest: %s != %s", afterDraft.Digest, firstPinned.Digest)
	}
	assertClosureRevisions(t, afterDraft, 1, 1, 1)

	secondSnapshot, secondReceipt, publishEntrypointErr := store.PublishEntrypoint(
		ctx, namespaceID, entrypointV1.ID, 3, mutationMeta("publish edited closure"),
	)
	if publishEntrypointErr != nil || secondReceipt.DesiredRevision != 2 {
		t.Fatalf("second PublishEntrypoint() = %+v, %v", secondReceipt, publishEntrypointErr)
	}
	assertClosureRevisions(t, secondSnapshot, 2, 2, 2)
	if secondSnapshot.Digest == firstSnapshot.Digest {
		t.Fatal("publishing a new exact closure retained the old digest")
	}
	assertOutboxCount(t, ctx, db, 2)

	if _, err := store.DeleteModel(ctx, namespaceID, modelV1.ID, 2, mutationMeta("delete Model draft root")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.DeleteRecipe(ctx, namespaceID, recipeV1.ID, 2, mutationMeta("delete Recipe draft root")); err != nil {
		t.Fatal(err)
	}
	afterDelete := loadPublishedSnapshot(t, ctx, db, namespaceID, secondReceipt.DesiredRevision)
	assertClosureRevisions(t, afterDelete, 2, 2, 2)
	assertOutboxCount(t, ctx, db, 2)
	if _, _, err := store.PublishEntrypoint(
		ctx, namespaceID, entrypointV1.ID, 4, mutationMeta("reject deleted dependencies"),
	); err == nil || !strings.Contains(err.Error(), routingmanagement.ErrPublication.Error()) {
		t.Fatalf("PublishEntrypoint() with deleted dependencies error = %v", err)
	}
	assertOutboxCount(t, ctx, db, 2)
}

func TestPublishRejectsSemanticRecipeErrorsBeforeCommit(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	if _, _, err := store.CreateModels(ctx, namespaceID,
		[]routingsnapshot.Model{routingTestModel(1, "provider-model")},
		mutationMeta("create Model")); err != nil {
		t.Fatal(err)
	}
	recipe := routingTestRecipe(1, "Simple")
	recipe.Document = []byte(`{
  "signals": {},
  "projections": {},
  "decisions": [{
    "name":"Simple",
    "rules":{"type":"metadata","name":"missing-metadata-signal"}
  }]
}`)
	if _, _, err := store.CreateRecipe(ctx, namespaceID, "", recipe, mutationMeta("create invalid Recipe draft")); err != nil {
		t.Fatal(err)
	}
	entrypoint := routingTestEntrypoint(1, 1, 1)
	if _, _, err := store.CreateEntrypoint(ctx, namespaceID, entrypoint, mutationMeta("create Entrypoint")); err != nil {
		t.Fatal(err)
	}

	if _, _, err := store.PublishEntrypoint(
		ctx, namespaceID, entrypoint.ID, 1, mutationMeta("publish invalid closure"),
	); err == nil || !strings.Contains(err.Error(), routingmanagement.ErrPublication.Error()) {
		t.Fatalf("PublishEntrypoint() error = %v", err)
	}
	stored, err := store.GetEntrypoint(ctx, namespaceID, entrypoint.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Status != routingmanagement.StatusDraft || stored.Revision != 1 {
		t.Fatalf("failed publication mutated Entrypoint = %#v", stored.ResourceIdentity)
	}
	assertOutboxCount(t, ctx, db, 0)
}

func TestModelUpdateCASHasOneWinner(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, _, err := store.CreateModels(ctx, namespaceID, []routingsnapshot.Model{routingTestModel(1, "base")}, mutationMeta("create Model")); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	errorsOut := make(chan error, 2)
	var wait sync.WaitGroup
	for index := 0; index < 2; index++ {
		wait.Add(1)
		go func(candidate int) {
			defer wait.Done()
			<-start
			_, _, err := store.UpdateModel(ctx, namespaceID, "model_one", 1,
				routingTestModel(2, fmt.Sprintf("candidate-%d", candidate)), mutationMeta("concurrent edit"))
			errorsOut <- err
		}(index)
	}
	close(start)
	wait.Wait()
	close(errorsOut)
	successes, conflicts := 0, 0
	for err := range errorsOut {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, routingmanagement.ErrConflict):
			conflicts++
		default:
			t.Fatalf("concurrent UpdateModel() error = %v", err)
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("CAS outcomes success=%d conflict=%d", successes, conflicts)
	}
}

func TestPriorityFallbackPublishesIntoTheImmutableSnapshot(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	primary := routingTestModel(1, "primary")
	backup := routingTestModel(1, "backup")
	backup.ID, backup.Name, backup.Backends[0].ID = "model_two", "Model Two", uuid.NewString()
	if _, _, err := store.CreateModels(ctx, namespaceID, []routingsnapshot.Model{primary, backup}, mutationMeta("create fallback Models")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.CreateRecipe(ctx, namespaceID, "", routingTestRecipe(1, "Simple"), mutationMeta("create Recipe")); err != nil {
		t.Fatal(err)
	}
	entrypoint := routingTestEntrypoint(1, 1, 1)
	entrypoint.Rules[0].Assignments["decision_simple"] = routingsnapshot.AssignmentSet{
		Models: []routingsnapshot.Assignment{
			{ModelID: primary.ID, ModelRevision: 1, Priority: 0, Weight: "1"},
			{ModelID: backup.ID, ModelRevision: 1, Priority: 1, Weight: "1"},
		},
		Fallback: &routingsnapshot.FallbackPolicy{Strategy: "priority", On: []string{"unavailable", "timeout"}},
	}
	created, _, err := store.CreateEntrypoint(ctx, namespaceID, entrypoint, mutationMeta("author fallback"))
	if err != nil {
		t.Fatal(err)
	}
	set := created.Current.Rules[0].Assignments["decision_simple"]
	if len(set.Models) != 2 || set.Models[1].Priority != 1 || set.Fallback == nil || len(set.Fallback.On) != 2 {
		t.Fatalf("stored fallback = %#v", set)
	}
	published, receipt, err := store.PublishEntrypoint(ctx, namespaceID, entrypoint.ID, 1, mutationMeta("publish fallback"))
	if err != nil {
		t.Fatalf("PublishEntrypoint() error = %v", err)
	}
	if receipt.DesiredRevision <= 0 || published == nil || len(published.Entrypoints) != 1 {
		t.Fatalf("published snapshot or receipt is incomplete: snapshot=%#v receipt=%#v", published, receipt)
	}
	publishedSet := published.Entrypoints[0].Rules[0].Assignments["decision_simple"]
	if len(publishedSet.Models) != 2 || publishedSet.Models[1].Priority != 1 ||
		publishedSet.Fallback == nil || publishedSet.Fallback.Strategy != "priority" ||
		len(publishedSet.Fallback.On) != 2 {
		t.Fatalf("published fallback = %#v", publishedSet)
	}
}

type builtInRecipeFixture struct {
	ctx              context.Context
	db               *sql.DB
	firstNamespaceID string
	firstStore       *Store
	secondStore      *Store
	distribution     routingmanagement.BuiltInRecipeDistribution
}

func TestBuiltInRecipeInstallationIsCrossReplicaNamespaceSafeAndImmutable(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fixture := builtInRecipeFixture{
		ctx: ctx, db: db, firstNamespaceID: namespaceID,
		firstStore: newRoutingStore(t, db), secondStore: newRoutingStore(t, db),
		distribution: routingTestBuiltInDistribution(t, "1.0.0", "Simple"),
	}
	assertConcurrentBuiltInInstallation(t, fixture)
	expected := assertBuiltInRecipeImmutability(t, fixture)
	assertBuiltInRecipeNamespaceAndUpgrade(t, fixture, expected)
}

func assertConcurrentBuiltInInstallation(t *testing.T, fixture builtInRecipeFixture) {
	t.Helper()
	start := make(chan struct{})
	errorsOut := make(chan error, 2)
	var wait sync.WaitGroup
	for index, store := range []*Store{fixture.firstStore, fixture.secondStore} {
		wait.Add(1)
		go func(replica int, active *Store) {
			defer wait.Done()
			<-start
			_, err := active.InstallBuiltInRecipes(
				fixture.ctx, fixture.firstNamespaceID, fixture.distribution,
				routingmanagement.MutationContext{
					RequestID: uuid.NewString(), Reason: fmt.Sprintf("replica %d install", replica),
				},
			)
			errorsOut <- err
		}(index, store)
	}
	close(start)
	wait.Wait()
	close(errorsOut)
	for err := range errorsOut {
		if err != nil {
			t.Fatalf("concurrent InstallBuiltInRecipes() error = %v", err)
		}
	}
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, fixture.firstNamespaceID, 1, 1)
	if err := fixture.firstStore.VerifyBuiltInRecipes(fixture.ctx, fixture.distribution); err != nil {
		t.Fatal(err)
	}
}

func assertBuiltInRecipeImmutability(
	t *testing.T,
	fixture builtInRecipeFixture,
) routingsnapshot.Recipe {
	t.Helper()
	expected, err := fixture.distribution.RecipesForNamespace(fixture.firstNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	installed, err := fixture.firstStore.GetRecipe(
		fixture.ctx, fixture.firstNamespaceID, expected[0].ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if installed.Origin != routingmanagement.RecipeOriginDistribution || installed.Provenance == nil ||
		installed.Provenance.DistributionID != fixture.distribution.ID ||
		installed.Provenance.DistributionVersion != fixture.distribution.Version ||
		installed.Provenance.SourceRecipeID != fixture.distribution.Recipes[0].SourceID {
		t.Fatalf("installed provenance = %#v", installed)
	}
	changed := expected[0]
	changed.Revision++
	changed.Name = "Changed"
	if _, _, err := fixture.firstStore.UpdateRecipe(
		fixture.ctx, fixture.firstNamespaceID, installed.ID, installed.Revision,
		"changed", changed, mutationMeta("reject built-in edit"),
	); !errors.Is(err, routingmanagement.ErrImmutable) {
		t.Fatalf("UpdateRecipe() error = %v, want ErrImmutable", err)
	}
	if _, err := fixture.firstStore.DeleteRecipe(
		fixture.ctx, fixture.firstNamespaceID, installed.ID, installed.Revision,
		mutationMeta("reject built-in delete"),
	); !errors.Is(err, routingmanagement.ErrImmutable) {
		t.Fatalf("DeleteRecipe() error = %v, want ErrImmutable", err)
	}
	return expected[0]
}

func assertBuiltInRecipeNamespaceAndUpgrade(
	t *testing.T,
	fixture builtInRecipeFixture,
	expected routingsnapshot.Recipe,
) {
	t.Helper()
	secondNamespaceID := insertRoutingNamespace(t, fixture.ctx, fixture.db)
	installer, err := routingmanagement.NewBuiltInRecipeInstaller(routingmanagement.BuiltInRecipeInstallerOptions{
		Store: fixture.firstStore, Distribution: fixture.distribution,
	})
	if err != nil {
		t.Fatal(err)
	}
	if reconcileErr := installer.Reconcile(fixture.ctx); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, secondNamespaceID, 1, 1)
	secondExpected, err := fixture.distribution.RecipesForNamespace(secondNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	if expected.ID == secondExpected[0].ID {
		t.Fatal("globally keyed Recipe identity collided across Namespaces")
	}
	restarted, err := routingmanagement.NewBuiltInRecipeInstaller(routingmanagement.BuiltInRecipeInstallerOptions{
		Store: fixture.secondStore, Distribution: fixture.distribution,
	})
	if err != nil {
		t.Fatal(err)
	}
	if reconcileErr := restarted.Reconcile(fixture.ctx); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, fixture.firstNamespaceID, 1, 1)
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, secondNamespaceID, 1, 1)
	upgrade := routingTestBuiltInDistribution(t, "1.1.0", "Simple v2")
	upgrader, err := routingmanagement.NewBuiltInRecipeInstaller(routingmanagement.BuiltInRecipeInstallerOptions{
		Store: fixture.firstStore, Distribution: upgrade,
	})
	if err != nil {
		t.Fatal(err)
	}
	if reconcileErr := upgrader.Reconcile(fixture.ctx); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, fixture.firstNamespaceID, 2, 2)
	assertBuiltInRecipeCounts(t, fixture.ctx, fixture.db, secondNamespaceID, 2, 2)
	changedInPlace := routingTestBuiltInDistribution(t, "1.0.0", "Changed in place")
	conflicting, err := routingmanagement.NewBuiltInRecipeInstaller(routingmanagement.BuiltInRecipeInstallerOptions{
		Store: fixture.firstStore, Distribution: changedInPlace,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := conflicting.Reconcile(fixture.ctx); !errors.Is(err, routingmanagement.ErrConflict) {
		t.Fatalf("changed-in-place reconciliation error = %v, want ErrConflict", err)
	}
	if _, err := fixture.db.ExecContext(fixture.ctx, `UPDATE routing_recipes SET description='tampered'
WHERE namespace_id=$1 AND id=$2`, fixture.firstNamespaceID, expected.ID); err != nil {
		t.Fatal(err)
	}
	if err := fixture.firstStore.VerifyBuiltInRecipes(fixture.ctx, fixture.distribution); !errors.Is(err, routingmanagement.ErrConflict) {
		t.Fatalf("immutable-state verification error = %v, want ErrConflict", err)
	}
}

func routingTestBuiltInDistribution(t *testing.T, version, decisionName string) routingmanagement.BuiltInRecipeDistribution {
	t.Helper()
	metadata := []byte("schema_version: vllm-sr/recipe-metadata/v1\nid: test-recipes\nname: Test Recipes\nversion: " + version + "\n")
	document := []byte("version: v0.3\nrecipes:\n" +
		"  - name: Simple\n" +
		"    description: A reusable test Recipe.\n" +
		"    routing:\n" +
		"      decisions:\n" +
		"        - name: " + decisionName + "\n" +
		"          rules: {}\n")
	distribution, err := routingmanagement.ParseBuiltInRecipeDistribution(metadata, document)
	if err != nil {
		t.Fatal(err)
	}
	return distribution
}

func insertRoutingNamespace(t *testing.T, ctx context.Context, db *sql.DB) string {
	t.Helper()
	namespaceID := uuid.NewString()
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, namespaceID, "namespace-"+namespaceID, "quota-"+namespaceID); err != nil {
		t.Fatal(err)
	}
	return namespaceID
}

func assertBuiltInRecipeCounts(
	t *testing.T, ctx context.Context, db *sql.DB, namespaceID string, recipes, distributions int,
) {
	t.Helper()
	var recipeCount, provenanceCount, distributionCount, auditCount, outboxCount int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM routing_recipes WHERE namespace_id=$1 AND deleted_at IS NULL),
  (SELECT count(*) FROM routing_recipe_provenance WHERE namespace_id=$1),
  (SELECT count(*) FROM routing_recipe_distributions WHERE namespace_id=$1),
  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1 AND action='routing.recipe.distribution.install'),
  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1)`, namespaceID).Scan(
		&recipeCount, &provenanceCount, &distributionCount, &auditCount, &outboxCount,
	); err != nil {
		t.Fatal(err)
	}
	if recipeCount != recipes || provenanceCount != recipes || auditCount != recipes ||
		distributionCount != distributions || outboxCount != 0 {
		t.Fatalf("built-in state = recipes %d provenance %d distributions %d audit %d outbox %d",
			recipeCount, provenanceCount, distributionCount, auditCount, outboxCount)
	}
}

func routingTestModel(revision int64, providerModelID string) routingsnapshot.Model {
	return routingsnapshot.Model{
		ID: "model_one", Revision: revision, CatalogRevision: routingCatalogRevision,
		Name: "Model One", Execution: routingsnapshot.ModelExecution{
			MaxRetries: 2, RequestTimeout: "30s", StreamTimeout: "2m",
		},
		Backends: []routingsnapshot.Backend{{
			ID: uuid.NewString(), ProviderID: "provider_one", WireFormat: "openai.chat.v1",
			Origin: "https://models.example.com/v1", ProviderModelID: providerModelID,
			Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "1",
		}},
	}
}

func routingTestRecipe(revision int64, decisionName string) routingsnapshot.Recipe {
	return routingsnapshot.Recipe{
		ID: "recipe_one", Revision: revision, Name: "Recipe One",
		Decisions: []routingsnapshot.Decision{{ID: "decision_simple", Name: decisionName, DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
		Document: []byte(fmt.Sprintf(
			`{"signals":{},"projections":{},"decisions":[{"name":%q,"rules":{}}]}`,
			decisionName,
		)),
	}
}

func routingTestEntrypoint(revision, recipeRevision, modelRevision int64) routingsnapshot.Entrypoint {
	return routingsnapshot.Entrypoint{
		ID: "entrypoint_one", Revision: revision, Name: "Entrypoint One", Aliases: []string{"vllm-sr/one"},
		Rules: []routingsnapshot.EntrypointRule{{
			ID: "rule_default", Name: "Default", RecipeID: "recipe_one", RecipeRevision: recipeRevision,
			Assignments: map[string]routingsnapshot.AssignmentSet{
				"decision_simple": {Models: []routingsnapshot.Assignment{{ModelID: "model_one", ModelRevision: modelRevision, Weight: "1"}}},
			},
		}},
	}
}

func mutationMeta(reason string) routingmanagement.MutationContext {
	return routingmanagement.MutationContext{RequestID: uuid.NewString(), Reason: reason}
}

func loadPublishedSnapshot(t *testing.T, ctx context.Context, db *sql.DB, namespaceID string, revision int64) *routingsnapshot.Snapshot {
	t.Helper()
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	bundle, err := LoadPublishedBundle(ctx, tx, namespaceID, "USD", revision)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := routingsnapshot.Compile(bundle)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func assertClosureRevisions(t *testing.T, snapshot *routingsnapshot.Snapshot, model, recipe, entrypoint int64) {
	t.Helper()
	if snapshot == nil || len(snapshot.Models) != 1 || len(snapshot.Recipes) != 1 || len(snapshot.Entrypoints) != 1 ||
		snapshot.Models[0].Revision != model || snapshot.Recipes[0].Revision != recipe || snapshot.Entrypoints[0].Revision != entrypoint {
		t.Fatalf("closure revisions = %+v", snapshot)
	}
}

func assertOutboxCount(t *testing.T, ctx context.Context, db *sql.DB, expected int) {
	t.Helper()
	var count int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != expected {
		t.Fatalf("outbox count = %d, want %d", count, expected)
	}
}

func routingIntegrationDatabase(t *testing.T) (*sql.DB, string) {
	t.Helper()
	dsn := os.Getenv("ROUTINGMANAGEMENT_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTINGMANAGEMENT_POSTGRES_DSN is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, routingIntegrationDatabaseErr := sql.Open("postgres", dsn)
	if routingIntegrationDatabaseErr != nil {
		t.Fatal(routingIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	schema := "routing_management_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, routingIntegrationDatabaseErr := url.Parse(dsn)
	if routingIntegrationDatabaseErr != nil {
		t.Fatal(routingIntegrationDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, routingIntegrationDatabaseErr := sql.Open("postgres", parsed.String())
	if routingIntegrationDatabaseErr != nil {
		t.Fatal(routingIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO provider_catalog_revisions
  (revision,snapshot_bytes,snapshot_digest,integration_references,catalog,
   required_wire_formats,required_credential_adapters,required_discovery_adapters)
VALUES ($1,'x',decode(repeat('aa',32),'hex'),'[]'::jsonb,'{}'::jsonb,
  '[]'::jsonb,'[]'::jsonb,'[]'::jsonb)`, routingCatalogRevision); err != nil {
		t.Fatal(err)
	}
	namespaceID := uuid.NewString()
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, namespaceID, "namespace-"+namespaceID, "quota-"+namespaceID); err != nil {
		t.Fatal(err)
	}
	return db, namespaceID
}
