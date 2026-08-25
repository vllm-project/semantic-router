package postgres

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestSeedEmptyStoreIsOneTimeAndRestartIdempotent(t *testing.T) {
	database, existingNamespace := routingIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, err := database.ExecContext(ctx, `DELETE FROM access_namespaces WHERE id=$1`, existingNamespace); err != nil {
		t.Fatal(err)
	}
	store, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	snapshot := bootstrapRoutingSnapshot(t)
	materializations := 0
	seeded, err := store.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		materializations++
		return nil, nil
	})
	if err != nil || !seeded || materializations != 1 {
		t.Fatalf("initial SeedEmptyStore() = seeded %t, materializations %d, err %v", seeded, materializations, err)
	}

	restarted, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	seeded, err = restarted.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		t.Fatal("restart attempted to resolve file-authored credentials")
		return nil, nil
	})
	if err != nil || seeded {
		t.Fatalf("restart SeedEmptyStore() = seeded %t, err %v", seeded, err)
	}

	var namespaces, models, recipes, entrypoints, revisions, outbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces),
  (SELECT count(*) FROM routing_models),
  (SELECT count(*) FROM routing_recipes),
  (SELECT count(*) FROM routing_entrypoints),
  (SELECT count(*) FROM policy_revisions),
  (SELECT count(*) FROM policy_outbox)`).Scan(
		&namespaces, &models, &recipes, &entrypoints, &revisions, &outbox,
	); err != nil {
		t.Fatal(err)
	}
	if namespaces != 1 || models != 1 || recipes != 1 || entrypoints != 1 || revisions != 1 || outbox != 1 {
		t.Fatalf("durable bootstrap counts = ns %d models %d recipes %d entrypoints %d revisions %d outbox %d",
			namespaces, models, recipes, entrypoints, revisions, outbox)
	}
}

func TestSeedEmptyStoreRollsBackPartialBootstrap(t *testing.T) {
	database, existingNamespace := routingIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, err := database.ExecContext(ctx, `DELETE FROM access_namespaces WHERE id=$1`, existingNamespace); err != nil {
		t.Fatal(err)
	}
	store, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	snapshot := bootstrapRoutingSnapshot(t)
	sentinel := errors.New("credential source unavailable")
	if seeded, seedErr := store.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		return nil, sentinel
	}); seeded || !errors.Is(seedErr, sentinel) {
		t.Fatalf("failed SeedEmptyStore() = seeded %t, err %v", seeded, seedErr)
	}
	var namespaces, models, revisions, audit, outbox int
	if queryErr := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces),
  (SELECT count(*) FROM routing_models),
  (SELECT count(*) FROM policy_revisions),
  (SELECT count(*) FROM access_audit_events),
  (SELECT count(*) FROM policy_outbox)`).Scan(
		&namespaces, &models, &revisions, &audit, &outbox,
	); queryErr != nil {
		t.Fatal(queryErr)
	}
	if namespaces != 0 || models != 0 || revisions != 0 || audit != 0 || outbox != 0 {
		t.Fatalf("failed bootstrap leaked state = ns %d models %d revisions %d audit %d outbox %d",
			namespaces, models, revisions, audit, outbox)
	}
	seeded, err := store.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		return nil, nil
	})
	if err != nil || !seeded {
		t.Fatalf("bootstrap retry = seeded %t, err %v", seeded, err)
	}
}

func TestSeedEmptyStorePersistsOnlyEncryptedProviderCredential(t *testing.T) {
	database, existingNamespace := routingIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, err := database.ExecContext(ctx, `DELETE FROM access_namespaces WHERE id=$1`, existingNamespace); err != nil {
		t.Fatal(err)
	}
	store, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	credentialID := uuid.NewString()
	snapshot := bootstrapRoutingSnapshotWithCredential(t, credentialID)
	seed, codec, plaintext := bootstrapProviderCredential(t, snapshot, credentialID)
	seeded, err := store.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		return []BootstrapProviderCredential{seed}, nil
	})
	if err != nil || !seeded {
		t.Fatalf("SeedEmptyStore() = seeded %t, err %v", seeded, err)
	}

	var ciphertext, nonce []byte
	var keyVersion string
	if queryErr := database.QueryRowContext(ctx, `SELECT secret_ciphertext, ciphertext_nonce, kek_version
FROM provider_credential_versions WHERE namespace_id=$1 AND provider_credential_id=$2`,
		snapshot.NamespaceID, credentialID,
	).Scan(&ciphertext, &nonce, &keyVersion); queryErr != nil {
		t.Fatal(queryErr)
	}
	if len(ciphertext) == 0 || len(nonce) == 0 || keyVersion == "" ||
		bytes.Equal(ciphertext, plaintext) || bytes.Contains(ciphertext, plaintext) {
		t.Fatalf("persisted provider credential envelope is not opaque: ciphertext=%d nonce=%d key=%q",
			len(ciphertext), len(nonce), keyVersion)
	}
	persistedVersion := seed.Version
	persistedVersion.Envelope = accesscredential.Envelope{
		KeyVersion: keyVersion, Nonce: nonce, Ciphertext: ciphertext,
	}
	opened, err := codec.OpenActive(
		seed.Credential, persistedVersion,
		seed.Credential.ProviderID, seed.Credential.NormalizedOrigin, seed.Version.NotBefore,
	)
	if err != nil || !bytes.Equal(opened, plaintext) {
		t.Fatalf("open persisted provider credential: plaintext match %t, err %v", bytes.Equal(opened, plaintext), err)
	}
	providercredential.Zero(opened)

	restarted, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	seeded, err = restarted.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		t.Fatal("restart attempted to resolve a file-authored provider credential")
		return nil, nil
	})
	if err != nil || seeded {
		t.Fatalf("restart SeedEmptyStore() = seeded %t, err %v", seeded, err)
	}
}

func TestSeedEmptyStoreRollsBackAfterEncryptedCredentialInsert(t *testing.T) {
	database, existingNamespace := routingIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, err := database.ExecContext(ctx, `DELETE FROM access_namespaces WHERE id=$1`, existingNamespace); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `CREATE FUNCTION reject_bootstrap_model() RETURNS trigger
LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'reject bootstrap model'; END $$;
CREATE TRIGGER reject_bootstrap_model BEFORE INSERT ON routing_models
FOR EACH ROW EXECUTE FUNCTION reject_bootstrap_model()`); err != nil {
		t.Fatal(err)
	}
	store, err := New(database, func(*routingsnapshot.Snapshot) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	credentialID := uuid.NewString()
	snapshot := bootstrapRoutingSnapshotWithCredential(t, credentialID)
	seed, _, _ := bootstrapProviderCredential(t, snapshot, credentialID)
	if seeded, err := store.SeedEmptyStore(ctx, snapshot, func() ([]BootstrapProviderCredential, error) {
		return []BootstrapProviderCredential{seed}, nil
	}); seeded || err == nil {
		t.Fatalf("failed SeedEmptyStore() = seeded %t, err %v", seeded, err)
	}

	var namespaces, credentials, versions, models, revisions, audit, outbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces),
  (SELECT count(*) FROM provider_credentials),
  (SELECT count(*) FROM provider_credential_versions),
  (SELECT count(*) FROM routing_models),
  (SELECT count(*) FROM policy_revisions),
  (SELECT count(*) FROM access_audit_events),
  (SELECT count(*) FROM policy_outbox)`).Scan(
		&namespaces, &credentials, &versions, &models, &revisions, &audit, &outbox,
	); err != nil {
		t.Fatal(err)
	}
	if namespaces != 0 || credentials != 0 || versions != 0 || models != 0 ||
		revisions != 0 || audit != 0 || outbox != 0 {
		t.Fatalf("failed bootstrap leaked state = ns %d credentials %d versions %d models %d revisions %d audit %d outbox %d",
			namespaces, credentials, versions, models, revisions, audit, outbox)
	}
}

func bootstrapRoutingSnapshot(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	model := routingTestModel(1, "provider/model")
	recipe := routingTestRecipe(1, "Simple")
	entrypoint := routingTestEntrypoint(1, recipe.Revision, model.Revision)
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: uuid.NewString(), Revision: 1, Currency: "USD",
		Models: []routingsnapshot.Model{model}, Recipes: []routingsnapshot.Recipe{recipe},
		Entrypoints: []routingsnapshot.Entrypoint{entrypoint},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func bootstrapRoutingSnapshotWithCredential(t *testing.T, credentialID string) *routingsnapshot.Snapshot {
	t.Helper()
	model := routingTestModel(1, "provider/model")
	model.Backends[0].ProviderCredentialID = credentialID
	recipe := routingTestRecipe(1, "Simple")
	entrypoint := routingTestEntrypoint(1, recipe.Revision, model.Revision)
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: uuid.NewString(), Revision: 1, Currency: "USD",
		Models: []routingsnapshot.Model{model}, Recipes: []routingsnapshot.Recipe{recipe},
		Entrypoints: []routingsnapshot.Entrypoint{entrypoint},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func bootstrapProviderCredential(
	t *testing.T,
	snapshot *routingsnapshot.Snapshot,
	credentialID string,
) (BootstrapProviderCredential, providercredential.Codec, []byte) {
	t.Helper()
	versionID := uuid.NewString()
	now := time.Date(2026, 8, 25, 0, 0, 0, 0, time.UTC)
	backend := snapshot.Models[0].Backends[0]
	credential := providercredential.Credential{
		ID: credentialID, NamespaceID: snapshot.NamespaceID, Name: "Bootstrap provider",
		ProviderID: backend.ProviderID, CredentialMode: providercredential.ModeRequired,
		CredentialAdapterID: "bearer", CatalogRevision: snapshot.Models[0].CatalogRevision,
		NormalizedOrigin: backend.Origin, Status: providercredential.StatusActive,
		ActiveVersionID: &versionID, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	codec := providercredential.Codec{Keyring: accesscredential.KEKKeyring{
		ActiveVersion: "bootstrap-provider-kek-v1",
		Keys: map[string][]byte{
			"bootstrap-provider-kek-v1": bytes.Repeat([]byte{0x5a}, 32),
		},
	}}
	plaintext := []byte("bootstrap-provider-secret-never-persisted")
	version, err := codec.Seal(credential, versionID, plaintext, now)
	if err != nil {
		t.Fatal(err)
	}
	return BootstrapProviderCredential{Credential: credential, Version: version}, codec, plaintext
}
