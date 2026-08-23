// Package backendresolver adapts durable ProviderCredential versions to the
// narrow credential interface used by BackendInvoker.
package backendresolver

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

// Loader returns ciphertext and metadata only to this in-process runtime seam.
// Implementations must never substitute the active version in LoadPinned.
type Loader interface {
	LoadActiveProviderCredential(context.Context, string) (providercredential.Credential, providercredential.Version, error)
	LoadPinnedProviderCredential(context.Context, string, string) (providercredential.Credential, providercredential.Version, error)
}

// Materializer converts one decrypted opaque secret into provider-specific
// request authentication. It must not retain the input byte slice.
type Materializer interface {
	Materialize([]byte) (backendinvoker.Credential, error)
}

// Registry selects a provider adapter without examining secret material.
type Registry interface {
	ForAdapter(string) (Materializer, error)
}

type Resolver struct {
	Loader   Loader
	Codec    providercredential.Codec
	Registry Registry
	Now      func() time.Time
}

func (r Resolver) Pin(
	ctx context.Context,
	credentialID string,
	providerID string,
	origin string,
) (string, error) {
	if err := r.validateDependencies(); err != nil {
		return "", err
	}
	credential, version, err := r.Loader.LoadActiveProviderCredential(ctx, credentialID)
	if err != nil {
		return "", fmt.Errorf("load active provider credential: %w", err)
	}
	secret, err := r.Codec.OpenActive(credential, version, providerID, origin, r.now())
	if err != nil {
		return "", err
	}
	defer providercredential.Zero(secret)
	if _, err := r.Registry.ForAdapter(credential.CredentialAdapterID); err != nil {
		return "", err
	}
	return version.ID, nil
}

func (r Resolver) ResolvePinned(
	ctx context.Context,
	credentialID string,
	versionID string,
	providerID string,
	origin string,
) (backendinvoker.Credential, error) {
	if err := r.validateDependencies(); err != nil {
		return backendinvoker.Credential{}, err
	}
	credential, version, err := r.Loader.LoadPinnedProviderCredential(ctx, credentialID, versionID)
	if err != nil {
		return backendinvoker.Credential{}, fmt.Errorf("load pinned provider credential: %w", err)
	}
	secret, err := r.Codec.OpenPinned(credential, version, providerID, origin, r.now())
	if err != nil {
		return backendinvoker.Credential{}, err
	}
	defer providercredential.Zero(secret)
	materializer, err := r.Registry.ForAdapter(credential.CredentialAdapterID)
	if err != nil {
		return backendinvoker.Credential{}, err
	}
	resolved, err := materializer.Materialize(secret)
	if err != nil {
		return backendinvoker.Credential{}, fmt.Errorf("materialize provider credential: %w", err)
	}
	resolved.Version = version.ID
	return resolved, nil
}

func (r Resolver) validateDependencies() error {
	if r.Loader == nil || r.Registry == nil {
		return fmt.Errorf("provider credential loader and adapter registry are required")
	}
	return nil
}

func (r Resolver) now() time.Time {
	if r.Now != nil {
		return r.Now().UTC()
	}
	return time.Now().UTC()
}

// HeaderMaterializer covers token providers whose complete secret is placed in
// exactly one approved request header. Complex cloud signing adapters implement
// Materializer directly.
type HeaderMaterializer struct {
	Header string
	Prefix string
	Extra  map[string]string
}

func (m HeaderMaterializer) Materialize(secret []byte) (backendinvoker.Credential, error) {
	if strings.TrimSpace(m.Header) == "" || len(secret) == 0 {
		return backendinvoker.Credential{}, fmt.Errorf("provider header and secret are required")
	}
	extra := make(map[string][]string, len(m.Extra))
	for key, value := range m.Extra {
		if strings.TrimSpace(key) == "" || strings.ContainsAny(key, "\r\n") || strings.ContainsAny(value, "\r\n") {
			return backendinvoker.Credential{}, fmt.Errorf("provider extra header is invalid")
		}
		extra[key] = []string{value}
	}
	return backendinvoker.Credential{
		Header: m.Header, Prefix: m.Prefix, Secret: string(secret), Extra: extra,
	}, nil
}
