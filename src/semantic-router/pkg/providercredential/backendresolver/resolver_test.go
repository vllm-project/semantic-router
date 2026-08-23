package backendresolver

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const (
	resolverNamespaceID  = "11111111-1111-4111-8111-111111111111"
	resolverCredentialID = "22222222-2222-4222-8222-222222222222"
	resolverOldVersionID = "33333333-3333-4333-8333-333333333333"
	resolverNewVersionID = "44444444-4444-4444-8444-444444444444"
)

var resolverNow = time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)

func TestResolverPinsAndResolvesExactRetiringVersion(t *testing.T) {
	codec := resolverCodec()
	credential := resolverCredential(resolverOldVersionID)
	oldVersion, err := codec.Seal(credential, resolverOldVersionID, []byte("old-secret"), resolverNow)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewStaticRegistry(map[string]Materializer{
		"bearer": HeaderMaterializer{Header: "Authorization", Prefix: "Bearer "},
	})
	if err != nil {
		t.Fatal(err)
	}
	loader := &loaderStub{credential: credential, active: oldVersion, pinned: oldVersion}
	resolver := Resolver{Loader: loader, Codec: codec, Registry: registry, Now: func() time.Time { return resolverNow }}

	pinned, err := resolver.Pin(context.Background(), resolverCredentialID, "openai", "https://api.example.com/v1")
	if err != nil || pinned != resolverOldVersionID {
		t.Fatalf("Pin() = %q, %v", pinned, err)
	}

	credential.ActiveVersionID = stringPointer(resolverNewVersionID)
	retireAt := resolverNow.Add(time.Minute)
	oldVersion.Status = providercredential.VersionRetiring
	oldVersion.ExpiresAt = &retireAt
	loader.credential = credential
	loader.pinned = oldVersion
	resolved, err := resolver.ResolvePinned(
		context.Background(), resolverCredentialID, resolverOldVersionID,
		"openai", "https://api.example.com/v1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Version != resolverOldVersionID || resolved.Secret != "old-secret" ||
		resolved.Header != "Authorization" {
		t.Fatalf("unexpected resolved credential: %#v", resolved)
	}
	if loader.pinnedVersion != resolverOldVersionID {
		t.Fatalf("loader was asked for %q", loader.pinnedVersion)
	}
}

func TestResolverFailsClosedOnProviderOrOriginMismatch(t *testing.T) {
	codec := resolverCodec()
	credential := resolverCredential(resolverOldVersionID)
	version, err := codec.Seal(credential, resolverOldVersionID, []byte("secret"), resolverNow)
	if err != nil {
		t.Fatal(err)
	}
	registry, _ := NewStaticRegistry(map[string]Materializer{
		"bearer": HeaderMaterializer{Header: "Authorization", Prefix: "Bearer "},
	})
	resolver := Resolver{
		Loader: &loaderStub{credential: credential, active: version, pinned: version},
		Codec:  codec, Registry: registry, Now: func() time.Time { return resolverNow },
	}
	if _, err := resolver.Pin(context.Background(), resolverCredentialID, "openai", "https://other.example.com"); !errors.Is(err, providercredential.ErrMismatch) {
		t.Fatalf("origin mismatch error = %v", err)
	}
	if _, err := resolver.Pin(context.Background(), resolverCredentialID, "anthropic", credential.NormalizedOrigin); !errors.Is(err, providercredential.ErrMismatch) {
		t.Fatalf("provider mismatch error = %v", err)
	}
}

func TestStaticRegistryCopiesCallerMap(t *testing.T) {
	adapters := map[string]Materializer{
		"x-api-key": HeaderMaterializer{Header: "X-API-Key"},
	}
	registry, err := NewStaticRegistry(adapters)
	if err != nil {
		t.Fatal(err)
	}
	delete(adapters, "x-api-key")
	if _, err := registry.ForAdapter("x-api-key"); err != nil {
		t.Fatalf("registry changed with caller map: %v", err)
	}
}

func TestBuiltinRegistryContainsMechanismsNotProviders(t *testing.T) {
	registry, err := BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	for _, adapterID := range []string{"bearer", "x-api-key", "api-key"} {
		if _, err := registry.ForAdapter(adapterID); err != nil {
			t.Fatalf("adapter %q: %v", adapterID, err)
		}
	}
	if _, err := registry.ForAdapter("openai"); err == nil {
		t.Fatal("product provider leaked into runtime credential adapters")
	}
}

type loaderStub struct {
	credential    providercredential.Credential
	active        providercredential.Version
	pinned        providercredential.Version
	pinnedVersion string
}

func (l *loaderStub) LoadActiveProviderCredential(
	context.Context,
	string,
) (providercredential.Credential, providercredential.Version, error) {
	return l.credential, l.active, nil
}

func (l *loaderStub) LoadPinnedProviderCredential(
	_ context.Context,
	_ string,
	versionID string,
) (providercredential.Credential, providercredential.Version, error) {
	l.pinnedVersion = versionID
	return l.credential, l.pinned, nil
}

func resolverCredential(activeVersion string) providercredential.Credential {
	return providercredential.Credential{
		ID: resolverCredentialID, NamespaceID: resolverNamespaceID, Name: "Primary",
		ProviderID: "openai", CredentialMode: providercredential.ModeRequired,
		CredentialAdapterID: "bearer",
		CatalogRevision:     "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		NormalizedOrigin:    "https://api.example.com/v1",
		Status:              providercredential.StatusActive, ActiveVersionID: &activeVersion, Revision: 1,
		CreatedAt: resolverNow, UpdatedAt: resolverNow,
	}
}

func resolverCodec() providercredential.Codec {
	return providercredential.Codec{Keyring: accesscredential.KEKKeyring{
		ActiveVersion: "provider-kek-v1",
		Keys: map[string][]byte{
			"provider-kek-v1": []byte(strings.Repeat("p", 32)),
		},
	}}
}

func stringPointer(value string) *string { return &value }
