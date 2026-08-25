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
	backendID           string
	modelName           string
	connectionOrdinal   int
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
	bootstrapSnapshot, credentialNames, err := compileDurableBootstrapSnapshot(cfg.RoutingSnapshot)
	if err != nil {
		return err
	}
	_, err = store.SeedEmptyStore(ctx, bootstrapSnapshot, func() ([]routingpostgres.BootstrapProviderCredential, error) {
		return materializeBootstrapCredentials(cfg, bootstrapSnapshot, credentialNames, codec, catalog)
	})
	return err
}

// compileDurableBootstrapSnapshot is the one file-authoring to durable-state
// identity boundary. Human-authored credential aliases stay readable in YAML
// and DSL documents; PostgreSQL and every published backend reference receive
// a deterministic UUID scoped to the public Namespace. Dynamic Management
// resources already carry native UUIDs and never pass through this bootstrap
// compiler.
func compileDurableBootstrapSnapshot(
	snapshot *routingsnapshot.Snapshot,
) (*routingsnapshot.Snapshot, map[string]string, error) {
	if snapshot == nil {
		return nil, nil, errors.New("routing bootstrap snapshot is required")
	}
	if _, err := uuid.Parse(snapshot.NamespaceID); err != nil {
		return nil, nil, errors.New("routing bootstrap Namespace ID must be a UUID")
	}

	bundle := snapshot.Bundle
	bundle.Models = append([]routingsnapshot.Model(nil), snapshot.Models...)
	credentialNames := make(map[string]string)
	for modelIndex := range bundle.Models {
		model := &bundle.Models[modelIndex]
		model.Backends = append([]routingsnapshot.Backend(nil), model.Backends...)
		for backendIndex := range model.Backends {
			backend := &model.Backends[backendIndex]
			name := backend.ProviderCredentialID
			if name == "" {
				continue
			}
			if err := providercredential.ValidateName(name); err != nil {
				return nil, nil, fmt.Errorf(
					"bootstrap backend %q provider credential alias %q is invalid: %w",
					backend.ID, name, err,
				)
			}
			credentialID := durableBootstrapCredentialID(snapshot.NamespaceID, name)
			if previous, collision := credentialNames[credentialID]; collision && previous != name {
				return nil, nil, errors.New("routing bootstrap provider credential identity collision")
			}
			credentialNames[credentialID] = name
			backend.ProviderCredentialID = credentialID
		}
	}
	compiled, err := routingsnapshot.Compile(bundle)
	if err != nil {
		return nil, nil, fmt.Errorf("compile durable routing bootstrap snapshot: %w", err)
	}
	return compiled, credentialNames, nil
}

func durableBootstrapCredentialID(namespaceID, name string) string {
	namespace := uuid.MustParse(namespaceID)
	return uuid.NewSHA1(
		namespace,
		[]byte("vllm-sr/provider-credential-bootstrap/v1\x00"+name),
	).String()
}

func materializeBootstrapCredentials(
	cfg *config.RouterConfig,
	snapshot *routingsnapshot.Snapshot,
	credentialNames map[string]string,
	codec providercredential.Codec,
	catalog *providercatalog.Snapshot,
) ([]routingpostgres.BootstrapProviderCredential, error) {
	bindings, err := bootstrapCredentialBindings(snapshot, credentialNames, catalog)
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
		name, mapped := credentialNames[id]
		if !mapped {
			return nil, errors.New("durable routing bootstrap credential mapping is incomplete")
		}
		definition, exists := cfg.BackendCredentials.File[name]
		if !exists {
			return nil, fmt.Errorf("bootstrap provider credential alias %q has no file secret source", name)
		}
		binding := bindings[id]
		seed, err := materializeBootstrapCredential(
			name, id, definition, binding, snapshot, codec, now,
		)
		if err != nil {
			return nil, err
		}
		result = append(result, seed)
	}
	return result, nil
}

func materializeBootstrapCredential(
	name string,
	credentialID string,
	definition config.BackendCredentialConfig,
	binding bootstrapCredentialBinding,
	snapshot *routingsnapshot.Snapshot,
	codec providercredential.Codec,
	now time.Time,
) (routingpostgres.BootstrapProviderCredential, error) {
	secret, err := readBootstrapCredentialSecret(definition)
	if err != nil {
		return routingpostgres.BootstrapProviderCredential{}, fmt.Errorf(
			"load bootstrap provider credential alias %q: %w", name, err,
		)
	}
	return sealBootstrapCredential(name, credentialID, definition, binding, snapshot, codec, now, secret)
}

func sealBootstrapCredential(
	name string,
	credentialID string,
	definition config.BackendCredentialConfig,
	binding bootstrapCredentialBinding,
	snapshot *routingsnapshot.Snapshot,
	codec providercredential.Codec,
	now time.Time,
	secret []byte,
) (routingpostgres.BootstrapProviderCredential, error) {
	defer providercredential.Zero(secret)
	if definition.CredentialAdapterID != binding.credentialAdapterID {
		return routingpostgres.BootstrapProviderCredential{}, fmt.Errorf(
			"bootstrap provider credential alias %q adapter %q does not match provider %q adapter %q",
			name, definition.CredentialAdapterID, binding.providerID, binding.credentialAdapterID,
		)
	}
	versionID := uuid.NewSHA1(
		uuid.NameSpaceOID,
		[]byte("vllm-sr/provider-credential-bootstrap-version/v1\x00"+credentialID+"\x00"+snapshot.SemanticDigest),
	).String()
	displayName := name
	if _, parseErr := uuid.Parse(name); parseErr == nil {
		displayName = readableBootstrapCredentialName(binding)
	}
	credential := providercredential.Credential{
		ID: credentialID, NamespaceID: snapshot.NamespaceID, Name: displayName,
		ProviderID: binding.providerID, CredentialMode: binding.credentialMode,
		CredentialAdapterID: binding.credentialAdapterID,
		CatalogRevision:     binding.catalogRevision, NormalizedOrigin: binding.origin,
		Status: providercredential.StatusActive, ActiveVersionID: &versionID,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	version, err := codec.Seal(credential, versionID, secret, now)
	if err != nil {
		return routingpostgres.BootstrapProviderCredential{}, fmt.Errorf(
			"seal bootstrap provider credential alias %q: %w", name, err,
		)
	}
	return routingpostgres.BootstrapProviderCredential{Credential: credential, Version: version}, nil
}

func bootstrapCredentialBindings(
	snapshot *routingsnapshot.Snapshot,
	credentialNames map[string]string,
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
		for backendIndex, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			name, mapped := credentialNames[backend.ProviderCredentialID]
			if !mapped {
				return nil, fmt.Errorf(
					"bootstrap backend %q references an unmapped provider credential",
					backend.ID,
				)
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
				backendID:           backend.ID,
				modelName:           model.Name,
				connectionOrdinal:   backendIndex + 1,
			}
			if previous, exists := bindings[backend.ProviderCredentialID]; exists {
				if !previous.compatible(binding) {
					return nil, fmt.Errorf(
						"bootstrap provider credential alias %q is reused by incompatible backends %q and %q: "+
							"provider, credential mode, adapter, origin, and catalog revision must match",
						name, previous.backendID, binding.backendID,
					)
				}
				continue
			}
			bindings[backend.ProviderCredentialID] = binding
		}
	}
	return bindings, nil
}

func readableBootstrapCredentialName(binding bootstrapCredentialBinding) string {
	name := fmt.Sprintf(
		"%s · %s · connection %d",
		binding.modelName, binding.providerID, binding.connectionOrdinal,
	)
	for len(name) > 256 {
		runes := []rune(name)
		if len(runes) <= 1 {
			return "Provider credential"
		}
		name = string(runes[:len(runes)-1])
	}
	return name
}

func (binding bootstrapCredentialBinding) compatible(other bootstrapCredentialBinding) bool {
	return binding.providerID == other.providerID &&
		binding.origin == other.origin &&
		binding.catalogRevision == other.catalogRevision &&
		binding.credentialMode == other.credentialMode &&
		binding.credentialAdapterID == other.credentialAdapterID
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
