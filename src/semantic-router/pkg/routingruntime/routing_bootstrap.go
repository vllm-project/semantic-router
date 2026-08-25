package routingruntime

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	routingpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type bootstrapCredentialBinding struct {
	providerID          string
	origin              string
	catalogRevision     string
	credentialMode      providercredential.Mode
	credentialAdapterID string
}

func seedDurableRoutingDatabase(
	ctx context.Context,
	database *sql.DB,
	cfg *config.RouterConfig,
	codec providercredential.Codec,
	catalog *providercatalog.Snapshot,
) error {
	if database == nil || cfg == nil || catalog == nil {
		return errors.New("durable routing bootstrap dependencies are required")
	}
	bootstrap := *cfg
	store, err := routingpostgres.New(database, func(snapshot *routingsnapshot.Snapshot) error {
		_, compileErr := config.CompileDurableRoutingSnapshot(&bootstrap, snapshot)
		return compileErr
	})
	if err != nil {
		return fmt.Errorf("compose durable routing bootstrap: %w", err)
	}
	if cfg.RoutingSnapshot == nil {
		namespaceID := uuid.NewSHA1(
			uuid.NameSpaceOID,
			[]byte("vllm-sr/durable-routing-bootstrap/v1\x00"+cfg.DocumentHash),
		).String()
		_, seedErr := store.SeedEmptyNamespace(ctx, namespaceID, cfg.BillingCurrency)
		return seedErr
	}
	_, err = store.SeedEmptyStore(ctx, cfg.RoutingSnapshot, func() ([]routingpostgres.BootstrapProviderCredential, error) {
		return materializeBootstrapCredentials(cfg, cfg.RoutingSnapshot, codec, catalog)
	})
	return err
}

func materializeBootstrapCredentials(
	cfg *config.RouterConfig,
	snapshot *routingsnapshot.Snapshot,
	codec providercredential.Codec,
	catalog *providercatalog.Snapshot,
) ([]routingpostgres.BootstrapProviderCredential, error) {
	bindings, err := bootstrapCredentialBindings(snapshot, catalog)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(bindings))
	for id := range bindings {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	result := make([]routingpostgres.BootstrapProviderCredential, 0, len(ids))
	now := time.Now().UTC()
	for _, id := range ids {
		definition, exists := cfg.BackendCredentials.File[id]
		if !exists {
			return nil, fmt.Errorf("bootstrap provider credential %q has no file secret source", id)
		}
		secret, err := readBootstrapCredentialSecret(definition)
		if err != nil {
			return nil, fmt.Errorf("load bootstrap provider credential %q: %w", id, err)
		}
		binding := bindings[id]
		if definition.CredentialAdapterID != binding.credentialAdapterID {
			providercredential.Zero(secret)
			return nil, fmt.Errorf(
				"bootstrap provider credential %q adapter does not match provider %q",
				id, binding.providerID,
			)
		}
		versionID := uuid.NewSHA1(
			uuid.NameSpaceOID,
			[]byte("vllm-sr/provider-credential-bootstrap-version/v1\x00"+id+"\x00"+snapshot.SemanticDigest),
		).String()
		credential := providercredential.Credential{
			ID: id, NamespaceID: snapshot.NamespaceID,
			Name:       "bootstrap-" + strings.ReplaceAll(id, "-", "")[:12],
			ProviderID: binding.providerID, CredentialMode: binding.credentialMode,
			CredentialAdapterID: binding.credentialAdapterID,
			CatalogRevision:     binding.catalogRevision, NormalizedOrigin: binding.origin,
			Status: providercredential.StatusActive, ActiveVersionID: &versionID,
			Revision: 1, CreatedAt: now, UpdatedAt: now,
		}
		version, sealErr := codec.Seal(credential, versionID, secret, now)
		providercredential.Zero(secret)
		if sealErr != nil {
			return nil, fmt.Errorf("seal bootstrap provider credential %q: %w", id, sealErr)
		}
		result = append(result, routingpostgres.BootstrapProviderCredential{
			Credential: credential, Version: version,
		})
	}
	return result, nil
}

func bootstrapCredentialBindings(
	snapshot *routingsnapshot.Snapshot,
	catalog *providercatalog.Snapshot,
) (map[string]bootstrapCredentialBinding, error) {
	bindings := make(map[string]bootstrapCredentialBinding)
	if snapshot == nil || catalog == nil {
		return bindings, nil
	}
	for _, model := range snapshot.Models {
		if model.CatalogRevision != catalog.Revision() {
			return nil, fmt.Errorf("bootstrap Model %q does not match the installed provider catalog revision", model.ID)
		}
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			provider, found := catalog.Get(backend.ProviderID)
			if !found || provider.Credential.Mode == providercatalog.CredentialNone ||
				provider.Credential.AdapterID == "" {
				return nil, fmt.Errorf(
					"bootstrap backend %q references an incompatible provider credential",
					backend.ID,
				)
			}
			binding := bootstrapCredentialBinding{
				providerID: backend.ProviderID, origin: backend.Origin,
				catalogRevision:     model.CatalogRevision,
				credentialMode:      providercredential.Mode(provider.Credential.Mode),
				credentialAdapterID: provider.Credential.AdapterID,
			}
			if previous, exists := bindings[backend.ProviderCredentialID]; exists && previous != binding {
				return nil, fmt.Errorf("bootstrap provider credential %q is reused across incompatible backends", backend.ProviderCredentialID)
			}
			bindings[backend.ProviderCredentialID] = binding
		}
	}
	return bindings, nil
}

func readBootstrapCredentialSecret(definition config.BackendCredentialConfig) ([]byte, error) {
	if definition.SecretValue != "" {
		if definition.SecretValue != strings.TrimSpace(definition.SecretValue) ||
			strings.ContainsAny(definition.SecretValue, "\r\n\x00") {
			return nil, errors.New("literal secret is not canonical")
		}
		return []byte(definition.SecretValue), nil
	}
	if definition.SecretFile == "" && definition.SecretEnv == "" {
		return nil, errors.New("secret source is not configured")
	}
	value, err := readScalarSecret(definition.SecretFile, definition.SecretEnv)
	if err != nil {
		return nil, err
	}
	return []byte(value), nil
}
