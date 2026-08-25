package postgres

import (
	"context"
	"database/sql"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestManifestCredentialNamesAreNamespaceScopedAndInactiveReferencesFail(t *testing.T) {
	database, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, database)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	credentialID := uuid.NewString()
	name := "Readable provider credential"
	insertInactiveManifestCredential(t, ctx, database, namespaceID, credentialID, name)

	ids, err := store.ProviderCredentialIDsByName(ctx, namespaceID, []string{name})
	if err != nil || ids[name] != credentialID {
		t.Fatalf("ProviderCredentialIDsByName() = %#v, %v", ids, err)
	}
	names, err := store.ProviderCredentialNamesByID(ctx, namespaceID, []string{credentialID})
	if err != nil || names[credentialID] != name {
		t.Fatalf("ProviderCredentialNamesByID() = %#v, %v", names, err)
	}
	if _, err := store.ProviderCredentialIDsByName(ctx, namespaceID, []string{"unknown credential"}); !errors.Is(err, routingmanagement.ErrManifest) {
		t.Fatalf("unknown ProviderCredential name error = %v", err)
	}
	otherNamespace := uuid.NewString()
	if _, err := database.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, otherNamespace, "namespace-"+otherNamespace, "quota-"+otherNamespace); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ProviderCredentialIDsByName(ctx, otherNamespace, []string{name}); !errors.Is(err, routingmanagement.ErrManifest) {
		t.Fatalf("cross-Namespace ProviderCredential name error = %v", err)
	}
	if _, err := store.ProviderCredentialNamesByID(ctx, otherNamespace, []string{credentialID}); !errors.Is(err, routingmanagement.ErrPublication) {
		t.Fatalf("cross-Namespace ProviderCredential identity error = %v", err)
	}

	model := routingTestModel(1, "provider/model")
	model.Backends[0].ProviderCredentialID = credentialID
	recipe := routingTestRecipe(1, "Simple")
	entrypoint := routingTestEntrypoint(1, recipe.Revision, model.Revision)
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: namespaceID, Revision: 1, Models: []routingsnapshot.Model{model},
		Recipes: []routingsnapshot.Recipe{recipe}, Entrypoints: []routingsnapshot.Entrypoint{entrypoint},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.PreviewManifest(ctx, namespaceID, 0, snapshot); !errors.Is(err, routingmanagement.ErrPublication) {
		t.Fatalf("inactive ProviderCredential validation error = %v", err)
	}
}

func insertInactiveManifestCredential(
	t *testing.T, ctx context.Context, database *sql.DB,
	namespaceID, credentialID, name string,
) {
	t.Helper()
	if _, err := database.ExecContext(ctx, `INSERT INTO provider_credentials
  (id,namespace_id,name,provider_id,credential_mode,credential_adapter_id,
   provider_catalog_revision,normalized_origin,status,revision,deleted_at)
VALUES ($1,$2,$3,'openai-compatible','optional','bearer',$4,
  'https://models.example','deleted',1,clock_timestamp())`,
		credentialID, namespaceID, name, routingCatalogRevision,
	); err != nil {
		t.Fatal(err)
	}
}
