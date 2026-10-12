package kvtransfer

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRedisOptionsFromConfigPreservesSeconds(t *testing.T) {
	cfg := &config.RedisConfig{}
	cfg.Connection.Host = "127.0.0.1"
	cfg.Connection.Port = 6379
	cfg.Connection.Timeout = 5

	opts := redisOptionsFromConfig(cfg)
	if opts.Timeout != 5*time.Second {
		t.Fatalf("Timeout = %v, want 5s (config timeout is seconds, not milliseconds)", opts.Timeout)
	}
	if opts.Timeout == 5*time.Millisecond {
		t.Fatal("Timeout must not shrink configured seconds to milliseconds")
	}
}

func TestValkeyOptionsFromConfigPreservesSeconds(t *testing.T) {
	cfg := &config.ValkeyConfig{}
	cfg.Connection.Host = "127.0.0.1"
	cfg.Connection.Port = 6379
	cfg.Connection.Timeout = 5

	opts := valkeyOptionsFromConfig(cfg)
	if opts.Timeout != 5*time.Second {
		t.Fatalf("Timeout = %v, want 5s (config timeout is seconds, not milliseconds)", opts.Timeout)
	}
	if opts.Timeout == 5*time.Millisecond {
		t.Fatal("Timeout must not shrink configured seconds to milliseconds")
	}
}

func TestConnectionTimeoutSecondsDefault(t *testing.T) {
	if got := connectionTimeoutSeconds(0); got != defaultAddressRegistryTimeout {
		t.Fatalf("connectionTimeoutSeconds(0) = %v, want %v", got, defaultAddressRegistryTimeout)
	}
	if got := connectionTimeoutSeconds(-1); got != defaultAddressRegistryTimeout {
		t.Fatalf("connectionTimeoutSeconds(-1) = %v, want %v", got, defaultAddressRegistryTimeout)
	}
}
