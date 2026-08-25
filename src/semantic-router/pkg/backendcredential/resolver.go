// Package backendcredential resolves file-authored backend secrets from named,
// operator-owned bootstrap references. Request data is deliberately absent
// from this contract so callers cannot select or supply an upstream secret.
package backendcredential

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

type materializedSecret struct {
	adapterID string
	secret    []byte
	version   string
}

// Resolver is an immutable process-local view of file-authored backend secrets.
// Secrets are materialized once during startup and are never exposed through
// configuration or management responses.
type Resolver struct {
	mu       sync.RWMutex
	secrets  map[string]materializedSecret
	registry backendresolver.StaticRegistry
	closed   bool
}

// String and GoString keep accidental diagnostics from rendering the
// process-local secret map.
func (r *Resolver) String() string {
	return "backendcredential.Resolver(redacted)"
}

func (r *Resolver) GoString() string {
	return r.String()
}

// NewResolver materializes all named file-authored secret references. A missing
// or empty secret fails startup rather than silently forwarding caller auth.
func NewResolver(definitions map[string]config.BackendCredentialConfig) (_ *Resolver, resultErr error) {
	registry, err := backendresolver.BuiltinRegistry()
	if err != nil {
		return nil, fmt.Errorf("compose backend credential adapters: %w", err)
	}
	resolver := &Resolver{secrets: make(map[string]materializedSecret, len(definitions)), registry: registry}
	defer func() {
		if resultErr != nil {
			_ = resolver.Close()
		}
	}()
	names := make([]string, 0, len(definitions))
	for name := range definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		definition := definitions[name]
		secret, err := readSecret(definition)
		if err != nil {
			return nil, fmt.Errorf("load backend credential %q: %w", name, err)
		}
		if secret == "" {
			return nil, fmt.Errorf("load backend credential %q: secret is empty", name)
		}
		if _, adapterErr := registry.ForAdapter(definition.CredentialAdapterID); adapterErr != nil {
			return nil, fmt.Errorf("load backend credential %q: %w", name, adapterErr)
		}
		secretBytes := []byte(secret)
		resolver.secrets[name] = materializedSecret{
			adapterID: definition.CredentialAdapterID,
			secret:    secretBytes,
			version:   fileCredentialVersion(name, definition.CredentialAdapterID, secretBytes),
		}
	}
	return resolver, nil
}

func readSecret(definition config.BackendCredentialConfig) (string, error) {
	if definition.SecretValue != "" {
		return definition.SecretValue, nil
	}
	if definition.SecretFile != "" {
		value, err := os.ReadFile(definition.SecretFile)
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(string(value)), nil
	}
	if definition.SecretEnv != "" {
		value, ok := os.LookupEnv(definition.SecretEnv)
		if !ok {
			return "", fmt.Errorf("environment variable %q is not set", definition.SecretEnv)
		}
		return strings.TrimSpace(value), nil
	}
	return "", fmt.Errorf("no secret reference configured")
}

func fileCredentialVersion(name, adapterID string, secret []byte) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("vllm-sr/file-provider-credential/v1\x00"))
	_, _ = digest.Write([]byte(name))
	_, _ = digest.Write([]byte{0})
	_, _ = digest.Write([]byte(adapterID))
	_, _ = digest.Write([]byte{0})
	_, _ = digest.Write(secret)
	return hex.EncodeToString(digest.Sum(nil))
}

func (r *Resolver) Pin(
	_ context.Context,
	publication backendinvoker.CredentialPublication,
	credentialID string,
	providerID string,
	origin string,
) (string, error) {
	if r == nil {
		return "", fmt.Errorf("backend credential resolver is unavailable")
	}
	if err := validateLookup(publication, credentialID, providerID, origin); err != nil {
		return "", err
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return "", fmt.Errorf("backend credential resolver is closed")
	}
	secret, ok := r.secrets[credentialID]
	if !ok {
		return "", fmt.Errorf("backend credential %q is not configured", credentialID)
	}
	return secret.version, nil
}

func (r *Resolver) ResolvePinned(
	_ context.Context,
	publication backendinvoker.CredentialPublication,
	credentialID string,
	versionID string,
	providerID string,
	origin string,
) (backendinvoker.Credential, error) {
	if r == nil {
		return backendinvoker.Credential{}, fmt.Errorf("backend credential resolver is unavailable")
	}
	if err := validateLookup(publication, credentialID, providerID, origin); err != nil {
		return backendinvoker.Credential{}, err
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return backendinvoker.Credential{}, fmt.Errorf("backend credential resolver is closed")
	}
	secret, ok := r.secrets[credentialID]
	if !ok || versionID == "" || versionID != secret.version {
		return backendinvoker.Credential{}, fmt.Errorf("backend credential %q version is unavailable", credentialID)
	}
	materializer, err := r.registry.ForAdapter(secret.adapterID)
	if err != nil {
		return backendinvoker.Credential{}, err
	}
	copy := append([]byte(nil), secret.secret...)
	defer providercredential.Zero(copy)
	credential, err := materializer.Materialize(copy)
	if err != nil {
		return backendinvoker.Credential{}, fmt.Errorf("materialize backend credential %q: %w", credentialID, err)
	}
	credential.Version = versionID
	return credential, nil
}

func validateLookup(
	publication backendinvoker.CredentialPublication,
	credentialID string,
	providerID string,
	origin string,
) error {
	if err := publication.Validate(); err != nil {
		return err
	}
	for label, value := range map[string]string{
		"credential": credentialID, "provider": providerID, "origin": origin,
	} {
		if value == "" || strings.TrimSpace(value) != value || strings.ContainsRune(value, 0) {
			return fmt.Errorf("backend %s identity is invalid", label)
		}
	}
	return nil
}

func (r *Resolver) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	for name, secret := range r.secrets {
		providercredential.Zero(secret.secret)
		delete(r.secrets, name)
	}
	r.closed = true
	return nil
}

// HeadersToStrip is the fixed caller-credential surface. These headers are
// always removed before dispatch, including for unauthenticated backends.
func HeadersToStrip() []string {
	return []string{
		"authorization",
		"proxy-authorization",
		"cookie",
		"set-cookie",
		"x-api-key",
		"api-key",
		"x-amz-security-token",
		"x-auth-token",
		"x-azure-api-key",
		"x-goog-api-key",
		"x-openai-api-key",
		"x-subscription-key",
		"x-user-openai-key",
		"x-user-anthropic-key",
		"x-user-azure-openai-key",
		"x-user-bedrock-key",
		"x-user-gemini-key",
		"x-user-vertex-ai-key",
		"x-user-minimax-key",
	}
}
