package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const routingBootstrapAdvisoryLock int64 = 0x56534c4c4d535242

// BootstrapProviderCredential is already encrypted. Secret resolution and
// sealing happen through BootstrapCredentialMaterializer only after the store
// has proven that it is empty and while the seed transaction remains open.
type BootstrapProviderCredential struct {
	Credential providercredential.Credential
	Version    providercredential.Version
}

type BootstrapCredentialMaterializer func() ([]BootstrapProviderCredential, error)

// SeedEmptyNamespace creates the one public authoring Namespace for an empty
// v0.3 bootstrap. It emits no routing revision; the first published Entrypoint
// becomes revision one through the ordinary Management transaction path.
func (store *Store) SeedEmptyNamespace(
	ctx context.Context,
	namespaceID, currency string,
) (bool, error) {
	if store == nil || store.db == nil || namespaceID == "" {
		return false, errors.New("routing Namespace bootstrap dependencies are required")
	}
	if currency == "" {
		currency = "USD"
	}
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return false, fmt.Errorf("begin routing Namespace bootstrap: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock($1)`, routingBootstrapAdvisoryLock); err != nil {
		return false, fmt.Errorf("lock routing Namespace bootstrap authority: %w", err)
	}
	var count int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM access_namespaces`).Scan(&count); err != nil {
		return false, fmt.Errorf("inspect durable Management bootstrap state: %w", err)
	}
	if count == 0 {
		if _, err := tx.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status,revision,runtime_epoch)
VALUES ($1,'default','default',$2,'active',1,1)`, namespaceID, currency); err != nil {
			return false, fmt.Errorf("seed routing Namespace: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("commit routing Namespace bootstrap: %w", err)
	}
	return count == 0, nil
}

// SeedEmptyStore installs the v0.3 file snapshot exactly once. PostgreSQL is
// the only authority after commit: subsequent starts do not merge or compare
// file routing or credentials. The Namespace, encrypted credential versions,
// desired routing resources, revision, audit record, and outbox wake-up share
// one serializable transaction.
func (store *Store) SeedEmptyStore(
	ctx context.Context,
	snapshot *routingsnapshot.Snapshot,
	materializeCredentials BootstrapCredentialMaterializer,
) (bool, error) {
	if store == nil || store.db == nil || snapshot == nil || materializeCredentials == nil {
		return false, errors.New("routing bootstrap dependencies are required")
	}
	verified, err := store.validateBootstrapSnapshot(snapshot)
	if err != nil {
		return false, err
	}

	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return false, fmt.Errorf("begin routing bootstrap: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	populated, err := bootstrapStoreIsPopulated(ctx, tx)
	if err != nil {
		return false, err
	}
	if populated {
		if err := tx.Commit(); err != nil {
			return false, fmt.Errorf("finish routing bootstrap inspection: %w", err)
		}
		return false, nil
	}
	if err := seedBootstrapResources(ctx, tx, verified, materializeCredentials); err != nil {
		return false, err
	}
	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("commit routing bootstrap: %w", err)
	}
	return true, nil
}

func (store *Store) validateBootstrapSnapshot(snapshot *routingsnapshot.Snapshot) (*routingsnapshot.Snapshot, error) {
	verified, err := routingsnapshot.Compile(snapshot.Bundle)
	if err != nil || verified.Digest != snapshot.Digest || verified.SemanticDigest != snapshot.SemanticDigest {
		return nil, errors.New("routing bootstrap snapshot is invalid")
	}
	if verified.Revision != 1 || len(verified.Entrypoints) == 0 {
		return nil, errors.New("routing bootstrap snapshot must contain revision one Entrypoints")
	}
	if err := store.validatePublication(verified); err != nil {
		return nil, fmt.Errorf("validate routing bootstrap snapshot: %w", err)
	}
	return verified, nil
}

func bootstrapStoreIsPopulated(ctx context.Context, tx *sql.Tx) (bool, error) {
	if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock($1)`, routingBootstrapAdvisoryLock); err != nil {
		return false, fmt.Errorf("lock routing bootstrap authority: %w", err)
	}
	var namespaceCount int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM access_namespaces`).Scan(&namespaceCount); err != nil {
		return false, fmt.Errorf("inspect durable Management bootstrap state: %w", err)
	}
	return namespaceCount != 0, nil
}

func seedBootstrapResources(
	ctx context.Context,
	tx *sql.Tx,
	snapshot *routingsnapshot.Snapshot,
	materializeCredentials BootstrapCredentialMaterializer,
) error {
	if err := seedBootstrapNamespace(ctx, tx, snapshot); err != nil {
		return err
	}
	if err := seedBootstrapCredentials(ctx, tx, snapshot, materializeCredentials); err != nil {
		return err
	}
	if err := seedBootstrapModels(ctx, tx, snapshot); err != nil {
		return err
	}
	if err := seedBootstrapRecipes(ctx, tx, snapshot); err != nil {
		return err
	}
	if err := seedBootstrapEntrypoints(ctx, tx, snapshot); err != nil {
		return err
	}
	return publishBootstrapMutation(ctx, tx, snapshot)
}

func seedBootstrapNamespace(ctx context.Context, tx *sql.Tx, snapshot *routingsnapshot.Snapshot) error {
	currency := snapshot.Currency
	if currency == "" {
		currency = "USD"
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status,revision,runtime_epoch)
VALUES ($1,'default','default',$2,'active',1,1)`, snapshot.NamespaceID, currency); err != nil {
		return fmt.Errorf("seed routing Namespace: %w", err)
	}
	return nil
}

func seedBootstrapCredentials(
	ctx context.Context,
	tx *sql.Tx,
	snapshot *routingsnapshot.Snapshot,
	materialize BootstrapCredentialMaterializer,
) error {
	credentials, err := materialize()
	if err != nil {
		return fmt.Errorf("materialize routing bootstrap credentials: %w", err)
	}
	if err := validateBootstrapProviderCredentials(snapshot, credentials); err != nil {
		return err
	}
	for index := range credentials {
		if err := insertBootstrapProviderCredential(ctx, tx, snapshot.NamespaceID, credentials[index]); err != nil {
			return err
		}
	}
	return nil
}

func seedBootstrapModels(ctx context.Context, tx *sql.Tx, snapshot *routingsnapshot.Snapshot) error {
	for _, model := range snapshot.Models {
		if err := insertModelRevision(ctx, tx, snapshot.NamespaceID, model, "", true); err != nil {
			return fmt.Errorf("seed routing Model %q: %w", model.Name, err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE routing_models SET status='active'
WHERE namespace_id=$1 AND id=$2`, snapshot.NamespaceID, model.ID); err != nil {
			return fmt.Errorf("activate seeded routing Model %q: %w", model.Name, err)
		}
	}
	return nil
}

func seedBootstrapRecipes(ctx context.Context, tx *sql.Tx, snapshot *routingsnapshot.Snapshot) error {
	for _, recipe := range snapshot.Recipes {
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipes
  (id,namespace_id,name,description,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'active',$5,1)`, recipe.ID, snapshot.NamespaceID,
			recipe.Name, recipe.Description, recipe.Revision); err != nil {
			return fmt.Errorf("seed routing Recipe %q: %w", recipe.Name, classifyWriteError(err))
		}
		if err := insertRecipeRevision(ctx, tx, recipe, ""); err != nil {
			return fmt.Errorf("seed routing Recipe %q: %w", recipe.Name, err)
		}
	}
	return nil
}

func seedBootstrapEntrypoints(ctx context.Context, tx *sql.Tx, snapshot *routingsnapshot.Snapshot) error {
	for _, entrypoint := range snapshot.Entrypoints {
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_entrypoints
  (id,namespace_id,name,aliases,status,current_revision,published_revision,revision)
VALUES ($1,$2,$3,$4,'active',$5,$5,1)`, entrypoint.ID, snapshot.NamespaceID,
			entrypoint.Name, mustJSON(entrypoint.Aliases), entrypoint.Revision); err != nil {
			return fmt.Errorf("seed routing Entrypoint %q: %w", entrypoint.Name, classifyWriteError(err))
		}
		if err := insertEntrypointRevision(ctx, tx, entrypoint, ""); err != nil {
			return fmt.Errorf("seed routing Entrypoint %q: %w", entrypoint.Name, err)
		}
	}
	return nil
}

func publishBootstrapMutation(ctx context.Context, tx *sql.Tx, snapshot *routingsnapshot.Snapshot) error {
	root := snapshot.Entrypoints[0]
	receipt, err := appendMutation(ctx, tx, snapshot.NamespaceID, mutationRecord{
		resourceType: "routing_entrypoint", resourceID: root.ID,
		resourceRevision: root.Revision, action: "routing.bootstrap", operation: "created",
	}, routingmanagement.MutationContext{
		RequestID: "routing-bootstrap:" + snapshot.Digest,
		Reason:    "Seed the initial durable routing state.",
	}, true)
	if err != nil {
		return fmt.Errorf("publish routing bootstrap: %w", err)
	}
	if receipt.DesiredRevision != snapshot.Revision {
		return errors.New("routing bootstrap desired revision diverged from the immutable snapshot")
	}
	return nil
}

type bootstrapCredentialBinding struct {
	providerID       string
	normalizedOrigin string
	catalogRevision  string
}

func validateBootstrapProviderCredentials(
	snapshot *routingsnapshot.Snapshot,
	credentials []BootstrapProviderCredential,
) error {
	references := make(map[string]bootstrapCredentialBinding)
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			binding := bootstrapCredentialBinding{
				providerID: backend.ProviderID, normalizedOrigin: backend.Origin,
				catalogRevision: model.CatalogRevision,
			}
			if previous, exists := references[backend.ProviderCredentialID]; exists && previous != binding {
				return errors.New("routing bootstrap provider credential has incompatible backend bindings")
			}
			references[backend.ProviderCredentialID] = binding
		}
	}
	if len(credentials) != len(references) {
		return errors.New("routing bootstrap provider credentials differ from routing references")
	}
	seen := make(map[string]struct{}, len(credentials))
	for _, seed := range credentials {
		credential := seed.Credential
		binding, referenced := references[credential.ID]
		if !referenced {
			return errors.New("routing bootstrap provider credential is not referenced")
		}
		if _, duplicate := seen[credential.ID]; duplicate {
			return errors.New("routing bootstrap provider credential is duplicated")
		}
		seen[credential.ID] = struct{}{}
		if credential.ProviderID != binding.providerID ||
			credential.NormalizedOrigin != binding.normalizedOrigin ||
			credential.CatalogRevision != binding.catalogRevision {
			return errors.New("routing bootstrap provider credential binding is invalid")
		}
	}
	return nil
}

func insertBootstrapProviderCredential(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	seed BootstrapProviderCredential,
) error {
	credential, version := seed.Credential, seed.Version
	if credential.NamespaceID != namespaceID || version.NamespaceID != namespaceID ||
		credential.Validate() != nil || version.Validate() != nil || credential.ActiveVersionID == nil ||
		*credential.ActiveVersionID != version.ID || version.CredentialID != credential.ID {
		return errors.New("routing bootstrap provider credential is invalid")
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO provider_credentials
  (id,namespace_id,name,provider_id,credential_mode,credential_adapter_id,
   provider_catalog_revision,normalized_origin,status,active_version_id,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'active',$9,1,$10,$10)`, credential.ID,
		credential.NamespaceID, credential.Name, credential.ProviderID, credential.CredentialMode,
		credential.CredentialAdapterID, credential.CatalogRevision, credential.NormalizedOrigin,
		version.ID, credential.CreatedAt); err != nil {
		return fmt.Errorf("seed provider credential metadata: %w", classifyWriteError(err))
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO provider_credential_versions
  (id,namespace_id,provider_credential_id,secret_ciphertext,ciphertext_nonce,kek_version,
   status,not_before,expires_at,revoked_at,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`, version.ID, version.NamespaceID,
		version.CredentialID, version.Envelope.Ciphertext, version.Envelope.Nonce,
		version.Envelope.KeyVersion, version.Status, version.NotBefore, version.ExpiresAt,
		version.RevokedAt, version.CreatedAt); err != nil {
		return fmt.Errorf("seed provider credential version: %w", classifyWriteError(err))
	}
	return nil
}
