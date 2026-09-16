package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRecoveryRedisOptionsUseDedicatedPrefix(t *testing.T) {
	routerConfig := &config.RouterConfig{}
	routerConfig.SemanticCache.Redis = &config.RedisConfig{}
	routerConfig.SemanticCache.Redis.Connection.Host = "redis.internal"
	routerConfig.SemanticCache.Redis.Connection.Port = 6380
	options, ok, err := recoveryRedisOptions(routerConfig, "redis", 1024)
	if err != nil || !ok {
		t.Fatalf("recoveryRedisOptions() = %#v, %v, %v", options, ok, err)
	}
	if options.Address != "redis.internal:6380" ||
		options.KeyPrefix != "vsr:context-recovery:v1:" ||
		options.MaxTotalBytes != 1024 {
		t.Fatalf("recovery options = %#v", options)
	}
}

func TestRecoveryTLSConfigRejectsIncompleteClientCertificate(t *testing.T) {
	if _, err := recoveryTLSConfig(
		"redis.internal",
		true,
		"client.crt",
		"",
		"",
	); err == nil {
		t.Fatal("recoveryTLSConfig() accepted incomplete client certificate")
	}
}

func TestContextCompressionTokenCalibrationCategoryUsesSelectedModel(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedModel:           "selected-model",
		RequestModel:               "auto",
		ContextCompressionRevision: "revision",
	}
	if got := contextCompressionTokenCalibrationCategory(ctx); got !=
		"context_compression:selected-model" {
		t.Fatalf("calibration category = %q", got)
	}
}
