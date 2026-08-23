package backendresolver

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

// PublishedLoader reads encrypted ProviderCredential material from the exact
// immutable publication named by a dispatch plan.
type PublishedLoader interface {
	LoadActivePublishedProviderCredential(
		context.Context, backendinvoker.CredentialPublication, string,
	) (providercredential.Credential, providercredential.Version, error)
	LoadPinnedPublishedProviderCredential(
		context.Context, backendinvoker.CredentialPublication, string, string,
	) (providercredential.Credential, providercredential.Version, error)
}

// PublishedResolver is the inference-only ProviderCredential seam. Unlike
// Resolver, its loader cannot address management storage and every lookup is
// pinned to a coupled access-and-routing publication.
type PublishedResolver struct {
	Loader   PublishedLoader
	Codec    providercredential.Codec
	Registry Registry
	Now      func() time.Time
}

func (r PublishedResolver) Pin(
	ctx context.Context,
	publication backendinvoker.CredentialPublication,
	credentialID string,
	providerID string,
	origin string,
) (string, error) {
	if err := r.validateDependencies(); err != nil {
		return "", err
	}
	if err := publication.Validate(); err != nil {
		return "", err
	}
	credential, version, err := r.Loader.LoadActivePublishedProviderCredential(ctx, publication, credentialID)
	if err != nil {
		return "", fmt.Errorf("load published active provider credential: %w", err)
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

func (r PublishedResolver) ResolvePinned(
	ctx context.Context,
	publication backendinvoker.CredentialPublication,
	credentialID string,
	versionID string,
	providerID string,
	origin string,
) (backendinvoker.Credential, error) {
	if err := r.validateDependencies(); err != nil {
		return backendinvoker.Credential{}, err
	}
	if err := publication.Validate(); err != nil {
		return backendinvoker.Credential{}, err
	}
	credential, version, err := r.Loader.LoadPinnedPublishedProviderCredential(
		ctx, publication, credentialID, versionID,
	)
	if err != nil {
		return backendinvoker.Credential{}, fmt.Errorf("load published pinned provider credential: %w", err)
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

func (r PublishedResolver) validateDependencies() error {
	if r.Loader == nil || r.Registry == nil {
		return fmt.Errorf("published provider credential loader and adapter registry are required")
	}
	return nil
}

func (r PublishedResolver) now() time.Time {
	if r.Now != nil {
		return r.Now().UTC()
	}
	return time.Now().UTC()
}

var _ backendinvoker.CredentialResolver = PublishedResolver{}
