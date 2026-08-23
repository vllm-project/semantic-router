package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

func TestRoutingListScopeUsesCompleteAuthorizedScope(t *testing.T) {
	namespaceID := "11111111-1111-4111-8111-111111111111"
	first := routingmanagement.ListQuery{Limit: 10, Scope: accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceModel: {"model_one"},
		},
	}}
	firstScope, err := normalizeListScope(namespaceID, accesscontrol.ScopeResourceModel, first)
	if err != nil {
		t.Fatal(err)
	}
	swapped := first
	swapped.Scope = accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceModel:  {"model_one"},
			accesscontrol.ScopeResourceRecipe: {"recipe_one"},
		},
	}
	swappedScope, err := normalizeListScope(namespaceID, accesscontrol.ScopeResourceModel, swapped)
	if err != nil {
		t.Fatal(err)
	}
	if firstScope.digest == swappedScope.digest {
		t.Fatal("distinct authorized scopes produced the same digest")
	}
}

func TestEmptyRoutingScopesReturnEmptyPagesWithoutOpeningTransaction(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	store := newRoutingStore(t, db)
	namespaceID := "11111111-1111-4111-8111-111111111111"
	request := routingmanagement.ListQuery{Limit: 50, Scope: accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		TeamIDs:     []accesscontrol.TeamID{"22222222-2222-4222-8222-222222222222"},
	}}
	models, err := store.ListModels(context.Background(), namespaceID, request)
	if err != nil || models.Items == nil || len(models.Items) != 0 {
		t.Fatalf("ListModels() = %#v, %v", models, err)
	}
	recipes, err := store.ListRecipes(context.Background(), namespaceID, request)
	if err != nil || recipes.Items == nil || len(recipes.Items) != 0 {
		t.Fatalf("ListRecipes() = %#v, %v", recipes, err)
	}
	entrypoints, err := store.ListEntrypoints(context.Background(), namespaceID, request)
	if err != nil || entrypoints.Items == nil || len(entrypoints.Items) != 0 {
		t.Fatalf("ListEntrypoints() = %#v, %v", entrypoints, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("empty result scope queried PostgreSQL: %v", err)
	}
}

func TestListEntrypointsLoadsSummariesInOneQuery(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	store := newRoutingStore(t, db)
	namespaceID := "11111111-1111-4111-8111-111111111111"
	createdAt, updatedAt := time.Unix(100, 0).UTC(), time.Unix(110, 0).UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(`(?s)WITH page AS .*count\(DISTINCT assignment\.model_id\).*ORDER BY page\.created_at DESC,page\.id DESC`).
		WithArgs(namespaceID, true, sqlmock.AnyArg(), routingmanagement.Status(""), nil, nil, 11).
		WillReturnRows(sqlmock.NewRows([]string{
			"namespace_id", "id", "name", "status", "revision", "created_at", "updated_at",
			"entrypoint_revision", "aliases", "rule_count", "assigned_model_count",
		}).AddRow(namespaceID, "entrypoint_one", "Entrypoint One", "active", 4,
			createdAt, updatedAt, 3, []byte(`["vllm-sr/one"]`), 2, 5))
	mock.ExpectCommit()

	page, err := store.ListEntrypoints(context.Background(), namespaceID, routingmanagement.ListQuery{
		Limit: 10, Scope: accesscontrol.ResultScope{
			NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 1 || page.Items[0].RuleCount != 2 || page.Items[0].AssignedModelCount != 5 ||
		len(page.Items[0].Current.Rules) != 0 {
		t.Fatalf("Entrypoint summaries = %#v", page.Items)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("Entrypoint list issued an unexpected topology query: %v", err)
	}
}

func TestListModelsLoadsOnePageInThreeQueries(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	store := newRoutingStore(t, db)
	namespaceID := "11111111-1111-4111-8111-111111111111"
	createdAt, updatedAt := time.Unix(100, 0).UTC(), time.Unix(110, 0).UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(`(?s)SELECT id FROM routing_models.*ORDER BY created_at DESC, id DESC LIMIT`).
		WithArgs(namespaceID, true, sqlmock.AnyArg(), routingmanagement.Status(""), nil, nil, 3).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow("model_two").AddRow("model_one"))
	mock.ExpectQuery(`(?s)SELECT m\.namespace_id,m\.id,r\.name.*m\.id=ANY`).
		WithArgs(namespaceID, sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{
			"namespace_id", "id", "name", "status", "revision", "created_at", "updated_at",
			"model_revision", "catalog_revision", "aliases", "param_size", "context_window_size",
			"description", "capabilities", "reasoning", "loras", "quality_score", "modality",
			"tags", "execution", "pricing",
		}).
			AddRow(namespaceID, "model_one", "Model One", "active", 1, createdAt, updatedAt,
				1, routingCatalogRevision, []byte(`[]`), "", 0, "", []byte(`[]`), []byte(`{}`),
				[]byte(`[]`), 0.0, "text", []byte(`[]`), []byte(`{}`), []byte(`{}`)).
			AddRow(namespaceID, "model_two", "Model Two", "active", 1, createdAt, updatedAt,
				1, routingCatalogRevision, []byte(`[]`), "", 0, "", []byte(`[]`), []byte(`{}`),
				[]byte(`[]`), 0.0, "text", []byte(`[]`), []byte(`{}`), []byte(`{}`)))
	mock.ExpectQuery(`(?s)SELECT b\.model_id,b\.id,b\.provider_id.*b\.model_revision=m\.current_revision`).
		WithArgs(namespaceID, sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{
			"model_id", "id", "provider_id", "wire_format", "normalized_origin",
			"provider_model_id", "provider_credential_id", "connection", "weight",
		}))
	mock.ExpectCommit()

	page, err := store.ListModels(context.Background(), namespaceID, routingmanagement.ListQuery{
		Limit: 2, Scope: accesscontrol.ResultScope{
			NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 2 || page.Items[0].ID != "model_two" || page.Items[1].ID != "model_one" {
		t.Fatalf("Model page order = %#v", page.Items)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("Model page issued an unexpected per-row query: %v", err)
	}
}

func TestListRecipesLoadsOnePageInThreeQueries(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	store := newRoutingStore(t, db)
	namespaceID := "11111111-1111-4111-8111-111111111111"
	createdAt, updatedAt := time.Unix(100, 0).UTC(), time.Unix(110, 0).UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(`(?s)SELECT id FROM routing_recipes.*ORDER BY created_at DESC,id DESC LIMIT`).
		WithArgs(namespaceID, true, sqlmock.AnyArg(), routingmanagement.Status(""), nil, nil, 3).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow("recipe_two").AddRow("recipe_one"))
	mock.ExpectQuery(`(?s)SELECT p\.namespace_id,p\.id,r\.name.*p\.id=ANY`).
		WithArgs(namespaceID, sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{
			"namespace_id", "id", "name", "description", "status", "revision", "created_at",
			"updated_at", "recipe_revision", "revision_description", "document", "distribution_id",
			"distribution_version", "asset_digest", "source_recipe_id", "source_recipe_revision",
			"recipe_digest", "installed_at",
		}).
			AddRow(namespaceID, "recipe_one", "Recipe One", "", "active", 1, createdAt, updatedAt,
				1, "", []byte(`{"decisions":[]}`), nil, nil, nil, nil, nil, nil, nil).
			AddRow(namespaceID, "recipe_two", "Recipe Two", "", "active", 1, createdAt, updatedAt,
				1, "", []byte(`{"decisions":[]}`), nil, nil, nil, nil, nil, nil, nil))
	mock.ExpectQuery(`(?s)SELECT d\.recipe_id,d\.decision_id,d\.name.*d\.recipe_revision=p\.current_revision`).
		WithArgs(namespaceID, sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{
			"recipe_id", "decision_id", "name", "dispatch_cardinality",
		}).AddRow("recipe_two", "decision_two", "Complex", "single"))
	mock.ExpectCommit()

	page, err := store.ListRecipes(context.Background(), namespaceID, routingmanagement.ListQuery{
		Limit: 2, Scope: accesscontrol.ResultScope{
			NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 2 || page.Items[0].ID != "recipe_two" || page.Items[1].ID != "recipe_one" ||
		len(page.Items[0].Current.Decisions) != 1 {
		t.Fatalf("Recipe page = %#v", page.Items)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("Recipe page issued an unexpected per-row query: %v", err)
	}
}

func TestRoutingListScopeRejectsWrongNamespace(t *testing.T) {
	request := routingmanagement.ListQuery{Limit: 50, Scope: accesscontrol.ResultScope{
		NamespaceID: "22222222-2222-4222-8222-222222222222",
		All:         true,
	}}
	_, err := normalizeListScope(
		"11111111-1111-4111-8111-111111111111",
		accesscontrol.ScopeResourceModel,
		request,
	)
	if !errors.Is(err, routingmanagement.ErrInvalid) {
		t.Fatalf("wrong namespace error = %v, want ErrInvalid", err)
	}
}
