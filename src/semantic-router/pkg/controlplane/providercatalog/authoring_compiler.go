package providercatalog

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// AuthoringCompiler compiles a concise file-authored Model connection through
// the same immutable Provider Integration registry used by dynamic authoring.
// Credential is a reference name; secret material is never loaded during
// compilation.
type AuthoringCompiler struct {
	Registry *Registry
}

func (compiler AuthoringCompiler) CompileConnection(
	_ context.Context,
	request modelauthoring.CompileRequest,
) (modelauthoring.CompileResult, error) {
	if compiler.Registry == nil {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider Integration registry is required")
	}
	connection := request.Connection
	if strings.TrimSpace(request.BackendID) == "" ||
		!idPattern.MatchString(connection.Provider) ||
		!canonicalText(connection.Model, 1, 512) {
		return modelauthoring.CompileResult{}, fmt.Errorf("backend identity, provider, and provider model are required")
	}
	snapshot := compiler.Registry.Snapshot()
	if snapshot == nil || !validCatalogRevision(snapshot.Revision()) {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider Integration catalog is unavailable")
	}
	provider, found := snapshot.Get(connection.Provider)
	if !found {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider Integration %q is not installed", connection.Provider)
	}
	providerInterface, compileConnectionErr := resolveInterface(provider, connection.Interface)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, compileConnectionErr
	}
	requestedEndpoint := connection.Endpoint
	if provider.Origin.Mode == OriginFixed && requestedEndpoint == provider.Origin.DefaultURL {
		requestedEndpoint = ""
	}
	origin, compileConnectionErr := providerOrigin(provider, requestedEndpoint)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, compileConnectionErr
	}
	if err := validateAuthoringCredential(provider, connection.Credential); err != nil {
		return modelauthoring.CompileResult{}, err
	}
	fields, compileConnectionErr := normalizeConnectionFields(provider.ConnectionFields, connection.ConnectionFields)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, compileConnectionErr
	}
	backendCompiler, found := compiler.Registry.BackendCompiler(providerInterface.Compiler.AdapterID)
	if !found {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider compiler %q is unavailable", providerInterface.Compiler.AdapterID)
	}
	compilerConfig, compileConnectionErr := cloneCompilerConfig(providerInterface.Compiler.Config)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider compiler config is invalid: %w", compileConnectionErr)
	}
	compiledConnection, compileConnectionErr := backendCompiler.Compile(compilerConfig, fields)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, fmt.Errorf("compile Provider connection: %w", compileConnectionErr)
	}
	compiledConnection.Path = compiledPathForOrigin(origin, compiledConnection.Path)
	compiledConnection, compileConnectionErr = applyTransportOverrides(compiledConnection, connection.Transport)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, fmt.Errorf("provider compiler emitted an invalid connection: %w", compileConnectionErr)
	}
	weight, compileConnectionErr := canonicalBackendWeight(connection.Weight)
	if compileConnectionErr != nil {
		return modelauthoring.CompileResult{}, fmt.Errorf("backend weight: %w", compileConnectionErr)
	}
	return modelauthoring.CompileResult{
		CatalogRevision: snapshot.Revision(),
		Backend: routingsnapshot.Backend{
			ID: request.BackendID, ProviderID: provider.ID,
			WireFormat: providerInterface.WireFormat, Origin: origin,
			ProviderModelID: connection.Model, ProviderCredentialID: connection.Credential,
			Connection: compiledConnection, Weight: weight,
		},
	}, nil
}

func applyTransportOverrides(
	compiled routingsnapshot.BackendConnection,
	overrides modelauthoring.TransportOverrides,
) (routingsnapshot.BackendConnection, error) {
	base, err := routingsnapshot.CanonicalizeBackendConnection(compiled)
	if err != nil {
		return routingsnapshot.BackendConnection{}, err
	}
	if overrides.Path != "" {
		base.Path = overrides.Path
	}
	override, err := routingsnapshot.CanonicalizeBackendConnection(routingsnapshot.BackendConnection{
		Path: base.Path, Headers: overrides.Headers,
	})
	if err != nil {
		return routingsnapshot.BackendConnection{}, fmt.Errorf("transport override: %w", err)
	}
	if len(override.Headers) > 0 {
		if base.Headers == nil {
			base.Headers = make(map[string]string, len(override.Headers))
		}
		for name, value := range override.Headers {
			base.Headers[name] = value
		}
	}
	return routingsnapshot.CanonicalizeBackendConnection(base)
}

func validateAuthoringCredential(provider Definition, credential string) error {
	if credential != strings.TrimSpace(credential) {
		return fmt.Errorf("credential reference must not contain surrounding whitespace")
	}
	switch provider.Credential.Mode {
	case CredentialNone:
		if credential != "" {
			return fmt.Errorf("credential is forbidden for Provider %q", provider.ID)
		}
	case CredentialRequired:
		if credential == "" {
			return fmt.Errorf("credential is required for Provider %q", provider.ID)
		}
	case CredentialOptional:
	default:
		return fmt.Errorf("provider %q has an invalid credential contract", provider.ID)
	}
	return nil
}

var _ modelauthoring.ConnectionCompiler = AuthoringCompiler{}
