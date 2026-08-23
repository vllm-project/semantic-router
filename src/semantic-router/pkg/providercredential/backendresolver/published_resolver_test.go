package backendresolver

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

func TestPublishedResolverPinsEveryLookupToPublicationIdentity(t *testing.T) {
	codec := resolverCodec()
	credential := resolverCredential(resolverOldVersionID)
	version, err := codec.Seal(credential, resolverOldVersionID, []byte("published-secret"), resolverNow)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewStaticRegistry(map[string]Materializer{
		"bearer": HeaderMaterializer{Header: "Authorization", Prefix: "Bearer "},
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := backendinvoker.CredentialPublication{
		NamespaceID: resolverNamespaceID, QuotaPartition: "quota-1", PublicationID: "pub_1",
	}
	loader := &publishedLoaderStub{credential: credential, active: version, pinned: version}
	resolver := PublishedResolver{
		Loader: loader, Codec: codec, Registry: registry, Now: func() time.Time { return resolverNow },
	}
	pinnedVersion, err := resolver.Pin(
		context.Background(), identity, resolverCredentialID, "openai", "https://api.example.com/v1",
	)
	if err != nil || pinnedVersion != resolverOldVersionID || loader.activeIdentity != identity {
		t.Fatalf("Pin() = %q, %v, identity=%+v", pinnedVersion, err, loader.activeIdentity)
	}
	resolved, err := resolver.ResolvePinned(
		context.Background(), identity, resolverCredentialID, resolverOldVersionID,
		"openai", "https://api.example.com/v1",
	)
	if err != nil || resolved.Secret != "published-secret" || resolved.Version != resolverOldVersionID ||
		loader.pinnedIdentity != identity || loader.pinnedVersion != resolverOldVersionID {
		t.Fatalf("ResolvePinned() = %+v, %v", resolved, err)
	}
}

type publishedLoaderStub struct {
	credential     providercredential.Credential
	active         providercredential.Version
	pinned         providercredential.Version
	activeIdentity backendinvoker.CredentialPublication
	pinnedIdentity backendinvoker.CredentialPublication
	pinnedVersion  string
}

func (loader *publishedLoaderStub) LoadActivePublishedProviderCredential(
	_ context.Context,
	identity backendinvoker.CredentialPublication,
	_ string,
) (providercredential.Credential, providercredential.Version, error) {
	loader.activeIdentity = identity
	return loader.credential, loader.active, nil
}

func (loader *publishedLoaderStub) LoadPinnedPublishedProviderCredential(
	_ context.Context,
	identity backendinvoker.CredentialPublication,
	_ string,
	versionID string,
) (providercredential.Credential, providercredential.Version, error) {
	loader.pinnedIdentity = identity
	loader.pinnedVersion = versionID
	return loader.credential, loader.pinned, nil
}
