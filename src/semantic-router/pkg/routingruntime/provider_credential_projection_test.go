package routingruntime

import (
	"testing"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

func TestDurableRoutingInferenceProviderCredentialResolverUsesOnlyPublicationStore(t *testing.T) {
	client := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1"})
	t.Cleanup(func() { _ = client.Close() })
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{Client: client})
	if err != nil {
		t.Fatal(err)
	}
	registry, err := backendresolver.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	resolver, err := composeInferenceProviderCredentialResolver(
		store, true, providercredential.Codec{}, registry,
	)
	if err != nil {
		t.Fatal(err)
	}
	if resolver.Loader != store {
		t.Fatalf("inference ProviderCredential loader = %T, want immutable publication store", resolver.Loader)
	}
	if _, ok := resolver.Loader.(*accesspublisher.RedisStore); !ok {
		t.Fatalf("inference ProviderCredential loader = %T, want Valkey publication store", resolver.Loader)
	}
	if _, err := composeInferenceProviderCredentialResolver(
		nil, true, providercredential.Codec{}, registry,
	); err == nil {
		t.Fatal("nil publication store unexpectedly composed")
	}
}
