package redis

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	redisclient "github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

func TestChallengeStoreIntegrationOneTimeAndRateLimited(t *testing.T) {
	address := os.Getenv("VLLM_SR_MANAGEMENT_AUTH_TEST_REDIS_URL")
	if address == "" {
		t.Skip("VLLM_SR_MANAGEMENT_AUTH_TEST_REDIS_URL is not configured")
	}
	options, err := redisclient.ParseURL(address)
	if err != nil {
		t.Fatal(err)
	}
	client := redisclient.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	prefix := "identity-test:" + uuid.NewString()
	t.Cleanup(func() {
		keys, _ := client.Keys(context.Background(), prefix+"*").Result()
		if len(keys) != 0 {
			_ = client.Del(context.Background(), keys...).Err()
		}
	})
	store, err := NewChallengeStore(ChallengeOptions{
		Client: client, KeyPrefix: prefix, TTL: 30 * time.Second, RateLimit: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	issuerID := uuid.NewString()
	challenge, err := store.Create(context.Background(), issuerID, "198.51.100.20", now)
	if err != nil {
		t.Fatal(err)
	}
	if challenge.Nonce == "" || !challenge.ExpiresAt.Equal(now.Add(30*time.Second)) {
		t.Fatalf("challenge = %+v", challenge)
	}
	if err := store.Consume(context.Background(), challenge.ID, issuerID, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", now); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("wrong nonce consume error = %v", err)
	}
	if err := store.Consume(context.Background(), challenge.ID, issuerID, challenge.Nonce, now); err != nil {
		t.Fatalf("consume error = %v", err)
	}
	if err := store.Consume(context.Background(), challenge.ID, issuerID, challenge.Nonce, now); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("replay error = %v", err)
	}
	if _, err := store.Create(context.Background(), issuerID, "198.51.100.20", now); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Create(context.Background(), issuerID, "198.51.100.20", now); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("rate limit error = %v", err)
	}
}
