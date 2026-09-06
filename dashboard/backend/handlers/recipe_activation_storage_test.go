package handlers

import (
	"reflect"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const storagePlannerCanonicalConfig = `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: test-model
  models:
    - name: test-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: general
        description: General requests
  decisions:
    - name: default
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: test-model
global: {}
`

func TestRequiredManagedStorageUsesCanonicalMemoryDefault(t *testing.T) {
	required, err := requiredManagedStorageBackends([]byte(storagePlannerCanonicalConfig))
	if err != nil {
		t.Fatalf("requiredManagedStorageBackends(): %v", err)
	}
	if len(required) != 0 {
		t.Fatalf("omitted local endpoints or memory response_cache unexpectedly require sidecars: %#v", required)
	}
}

func TestRequiredManagedStorageDerivesExactEffectiveManagedSet(t *testing.T) {
	t.Setenv("VLLM_SR_STACK_NAME", "lane-a")
	tests := []struct {
		name   string
		mutate func(*routerconfig.RouterConfig)
		want   map[string]struct{}
	}{
		{
			name: "memory cache does not create sidecars",
			mutate: func(config *routerconfig.RouterConfig) {
				config.SemanticCache.Enabled = true
				config.SemanticCache.BackendType = "memory"
			},
			want: map[string]struct{}{},
		},
		{
			name: "managed Redis cache ignores external durable services",
			mutate: func(config *routerconfig.RouterConfig) {
				config.SemanticCache.Enabled = true
				config.SemanticCache.BackendType = "redis"
				config.SemanticCache.Redis = &routerconfig.RedisConfig{}
				config.SemanticCache.Redis.Connection.Host = "lane-a-vllm-sr-redis"
				config.ResponseAPI.Redis.Address = "redis.external.example:6379"
				config.RouterReplay.Postgres = &routerconfig.RouterReplayPostgresConfig{Host: "postgres.external.example"}
			},
			want: map[string]struct{}{"redis": {}},
		},
		{
			name: "external Redis cache does not create a local sidecar",
			mutate: func(config *routerconfig.RouterConfig) {
				config.ResponseAPI.Enabled = false
				config.RouterReplay.Enabled = false
				config.SemanticCache.Enabled = true
				config.SemanticCache.BackendType = "redis"
				config.SemanticCache.Redis = &routerconfig.RedisConfig{}
				config.SemanticCache.Redis.Connection.Host = "redis.external.example"
			},
			want: map[string]struct{}{},
		},
		{
			name: "managed Milvus cache ignores external Redis and Postgres",
			mutate: func(config *routerconfig.RouterConfig) {
				config.SemanticCache.Enabled = true
				config.SemanticCache.BackendType = "milvus"
				config.SemanticCache.Milvus = &routerconfig.MilvusConfig{}
				config.SemanticCache.Milvus.Connection.Host = "milvus"
				config.ResponseAPI.Redis.Address = "redis.external.example:6379"
				config.RouterReplay.Postgres = &routerconfig.RouterReplayPostgresConfig{Host: "postgres.external.example"}
			},
			want: map[string]struct{}{"milvus": {}},
		},
		{
			name: "external Milvus cache does not create a local sidecar",
			mutate: func(config *routerconfig.RouterConfig) {
				config.ResponseAPI.Enabled = false
				config.RouterReplay.Enabled = false
				config.SemanticCache.Enabled = true
				config.SemanticCache.BackendType = "milvus"
				config.SemanticCache.Milvus = &routerconfig.MilvusConfig{}
				config.SemanticCache.Milvus.Connection.Host = "milvus.external.example"
			},
			want: map[string]struct{}{},
		},
		{
			name: "disabled managed cache does not create Milvus",
			mutate: func(config *routerconfig.RouterConfig) {
				config.SemanticCache.Enabled = false
				config.SemanticCache.BackendType = "milvus"
				config.SemanticCache.Milvus = &routerconfig.MilvusConfig{}
				config.SemanticCache.Milvus.Connection.Host = "lane-a-vllm-sr-milvus"
			},
			want: map[string]struct{}{},
		},
		{
			name: "enabled services and vector metadata use only local endpoints",
			mutate: func(config *routerconfig.RouterConfig) {
				config.ResponseAPI.Redis.Address = "lane-a-vllm-sr-redis:6379"
				config.RouterReplay.Enabled = true
				config.RouterReplay.StoreBackend = "postgres"
				config.RouterReplay.Postgres = &routerconfig.RouterReplayPostgresConfig{Host: "postgres"}
				config.VectorStore = &routerconfig.VectorStoreConfig{
					Enabled:       true,
					BackendType:   "milvus",
					Milvus:        &routerconfig.MilvusConfig{},
					MetadataStore: "postgres",
					MetadataPostgres: &routerconfig.VectorStoreMetadataPostgresConfig{
						Host: "postgres.external.example",
					},
				}
				config.VectorStore.Milvus.Connection.Host = "milvus.external.example"
			},
			want: map[string]struct{}{"redis": {}, "postgres": {}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertRequiredManagedStorage(t, test.mutate, test.want)
		})
	}
}

func assertRequiredManagedStorage(t *testing.T, mutate func(*routerconfig.RouterConfig), want map[string]struct{}) {
	t.Helper()
	config := routerconfig.DefaultGlobalConfig()
	mutate(&config)
	got := requiredManagedStorageForConfig(&config)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("required managed storage = %#v, want %#v", got, want)
	}
}

func TestManagedStorageEndpointRequiresThisStack(t *testing.T) {
	t.Setenv("VLLM_SR_STACK_NAME", "lane-a")
	for endpoint, want := range map[string]bool{
		"redis:6379":                     true,
		"lane-a-vllm-sr-redis:6379":      true,
		"redis.external.example:6379":    false,
		"redis-malicious.example:6379":   false,
		"redis://lane-a-vllm-sr-redis:1": true,
		"":                               false,
	} {
		if got := isManagedStorageEndpoint(endpoint, "redis"); got != want {
			t.Fatalf("isManagedStorageEndpoint(%q) = %v, want %v", endpoint, got, want)
		}
	}
}
