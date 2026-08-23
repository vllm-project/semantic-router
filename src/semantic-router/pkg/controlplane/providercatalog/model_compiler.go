package providercatalog

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var backendWeightPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?$`)

// ProviderCredentialReader is the non-secret metadata seam used while a Model
// is compiled. The compiler never loads or decrypts a credential version.
type ProviderCredentialReader interface {
	GetProviderCredential(context.Context, accesscontrol.NamespaceID, string) (providercredential.Credential, error)
}

// ModelCompiler turns the short control-plane Provider form into the complete,
// immutable Backend value consumed by routing publication. Product-specific
// defaults never cross this boundary as inference-runtime branching.
type ModelCompiler struct {
	Catalog     SnapshotSource
	Registry    *Registry
	Credentials ProviderCredentialReader
	Egress      backendegress.Policy
}

type ModelBackendRequest struct {
	NamespaceID      string
	BackendID        string
	ProviderID       string
	InterfaceID      string
	ProviderModelID  string
	CredentialID     string
	Origin           string
	ConnectionFields map[string]any
	Weight           string
}

type CompiledModelBackend struct {
	CatalogRevision string
	// ConnectionDigest is control-plane evidence that discovery and Model
	// compilation used the same normalized, non-secret Provider form.
	ConnectionDigest string
	Backend          routingsnapshot.Backend
}

func (compiler ModelCompiler) CompileBackend(
	ctx context.Context,
	request ModelBackendRequest,
) (CompiledModelBackend, error) {
	if compiler.Catalog == nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: active provider catalog source is required", ErrInvalidRequest)
	}
	if !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.BackendID) {
		return CompiledModelBackend{}, fmt.Errorf("%w: namespace and backend IDs must be canonical UUIDs", ErrInvalidRequest)
	}
	if !idPattern.MatchString(request.ProviderID) || !canonicalText(request.ProviderModelID, 1, 512) {
		return CompiledModelBackend{}, fmt.Errorf("%w: provider or provider model ID is invalid", ErrInvalidRequest)
	}

	snapshot, compileBackendErr := compiler.Catalog.ActiveSnapshot(ctx)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, fmt.Errorf("read active provider catalog snapshot: %w", compileBackendErr)
	}
	if snapshot == nil || !validCatalogRevision(snapshot.Revision()) {
		return CompiledModelBackend{}, fmt.Errorf("%w: active provider catalog snapshot is unavailable", ErrInvalidRequest)
	}
	provider, found := snapshot.Get(request.ProviderID)
	if !found {
		return CompiledModelBackend{}, ErrNotFound
	}
	providerInterface, compileBackendErr := resolveInterface(provider, request.InterfaceID)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, compileBackendErr
	}

	origin, compileBackendErr := providerOrigin(provider, request.Origin)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, compileBackendErr
	}
	if _, err := compiler.Egress.AuthorizeOrigin(origin); err != nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: provider origin is denied by backend egress policy", ErrInvalidRequest)
	}
	fields, compileBackendErr := normalizeConnectionFields(provider.ConnectionFields, request.ConnectionFields)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, compileBackendErr
	}
	connectionDigest, compileBackendErr := CanonicalConnectionDigest(fields)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, compileBackendErr
	}
	credentialID, compileBackendErr := compiler.validateCredential(ctx, snapshot, provider, request, origin)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, compileBackendErr
	}
	weight, compileBackendErr := canonicalBackendWeight(request.Weight)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: backend weight %w", ErrInvalidRequest, compileBackendErr)
	}

	if compiler.Registry == nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: Provider compiler registry is unavailable", ErrInvalidRequest)
	}
	backendCompiler, found := compiler.Registry.BackendCompiler(providerInterface.Compiler.AdapterID)
	if !found {
		return CompiledModelBackend{}, fmt.Errorf("%w: Provider compiler %q is unavailable", ErrInvalidRequest, providerInterface.Compiler.AdapterID)
	}
	compilerConfig, compileBackendErr := cloneCompilerConfig(providerInterface.Compiler.Config)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: Provider compiler config is invalid: %w", ErrInvalidRequest, compileBackendErr)
	}
	connection, compileBackendErr := backendCompiler.Compile(compilerConfig, cloneCanonicalConnectionValues(fields))
	if compileBackendErr != nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: compile Provider backend: %w", ErrInvalidRequest, compileBackendErr)
	}
	connection, compileBackendErr = routingsnapshot.CanonicalizeBackendConnection(connection)
	if compileBackendErr != nil {
		return CompiledModelBackend{}, fmt.Errorf("%w: Provider compiler emitted an invalid backend connection: %w", ErrInvalidRequest, compileBackendErr)
	}
	return CompiledModelBackend{
		CatalogRevision: snapshot.Revision(), ConnectionDigest: connectionDigest,
		Backend: routingsnapshot.Backend{
			ID: request.BackendID, ProviderID: provider.ID,
			WireFormat: providerInterface.WireFormat, Origin: origin,
			ProviderModelID: request.ProviderModelID, ProviderCredentialID: credentialID,
			Connection: connection,
			Weight:     weight,
		},
	}, nil
}

func resolveInterface(provider Definition, requested string) (Interface, error) {
	for _, candidate := range provider.Interfaces {
		if requested != "" && candidate.ID == requested || requested == "" && candidate.Default {
			return candidate, nil
		}
	}
	if requested == "" {
		return Interface{}, fmt.Errorf("%w: Provider %q has no default interface", ErrInvalidRequest, provider.ID)
	}
	return Interface{}, fmt.Errorf("%w: Provider interface %q is unavailable", ErrInvalidRequest, requested)
}

func (compiler ModelCompiler) validateCredential(
	ctx context.Context,
	snapshot *Snapshot,
	provider Definition,
	request ModelBackendRequest,
	origin string,
) (string, error) {
	switch provider.Credential.Mode {
	case CredentialNone:
		if request.CredentialID != "" {
			return "", fmt.Errorf("%w: credential is forbidden for this provider", ErrInvalidRequest)
		}
		return "", nil
	case CredentialRequired:
		if request.CredentialID == "" {
			return "", fmt.Errorf("%w: credential is required for this provider", ErrInvalidRequest)
		}
	case CredentialOptional:
		if request.CredentialID == "" {
			return "", nil
		}
	default:
		return "", fmt.Errorf("%w: provider credential mode is invalid", ErrInvalidRequest)
	}
	if !canonicalUUID(request.CredentialID) || compiler.Credentials == nil {
		return "", fmt.Errorf("%w: provider credential is invalid or its store is unavailable", ErrInvalidRequest)
	}
	credential, err := compiler.Credentials.GetProviderCredential(
		ctx, accesscontrol.NamespaceID(request.NamespaceID), request.CredentialID,
	)
	if err != nil {
		return "", fmt.Errorf("load provider credential metadata: %w", err)
	}
	if err := credential.Validate(); err != nil {
		return "", fmt.Errorf("%w: provider credential metadata is invalid", ErrInvalidRequest)
	}
	// A catalog-wide revision may change because an unrelated integration changed.
	// The immutable security binding is therefore the exact provider, adapter,
	// and canonical origin recorded with the credential. Its original catalog
	// revision remains provenance and encryption AAD, not a global equality
	// requirement for every future compatible Model revision.
	if credential.NamespaceID != request.NamespaceID || credential.Status != providercredential.StatusActive ||
		credential.ProviderID != provider.ID ||
		credential.CredentialMode != providercredential.Mode(provider.Credential.Mode) ||
		credential.CredentialAdapterID != provider.Credential.AdapterID ||
		credential.NormalizedOrigin != origin || !validCatalogRevision(credential.CatalogRevision) ||
		snapshot.Revision() == "" {
		return "", fmt.Errorf("%w: provider credential does not match the selected Provider binding", ErrInvalidRequest)
	}
	return credential.ID, nil
}

func providerOrigin(provider Definition, requested string) (string, error) {
	switch provider.Origin.Mode {
	case OriginFixed:
		if requested != "" {
			return "", fmt.Errorf("%w: fixed provider origin cannot be overridden", ErrInvalidRequest)
		}
		return provider.Origin.DefaultURL, nil
	case OriginUserSupplied:
		normalized, err := providercredential.NormalizeOrigin(requested)
		if err != nil {
			return "", fmt.Errorf("%w: provider origin is invalid", ErrInvalidRequest)
		}
		return normalized, nil
	default:
		return "", fmt.Errorf("%w: provider origin mode is invalid", ErrInvalidRequest)
	}
}

func canonicalBackendWeight(raw string) (string, error) {
	if raw == "" {
		return "1", nil
	}
	if !backendWeightPattern.MatchString(raw) {
		return "", errors.New("must be a plain positive decimal")
	}
	parts := strings.SplitN(raw, ".", 2)
	whole := strings.TrimLeft(parts[0], "0")
	if whole == "" {
		whole = "0"
	}
	canonical := whole
	if len(parts) == 2 {
		if len(parts[1]) > 9 {
			return "", errors.New("supports at most 9 fractional digits")
		}
		fraction := strings.TrimRight(parts[1], "0")
		if fraction != "" {
			canonical += "." + fraction
		}
	}
	if canonical == "0" {
		return "", errors.New("must be greater than zero")
	}
	return canonical, nil
}

func cloneStringMap(source map[string]string) map[string]string {
	if len(source) == 0 {
		return nil
	}
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
