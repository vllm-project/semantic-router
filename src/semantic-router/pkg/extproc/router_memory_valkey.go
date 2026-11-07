package extproc

import (
	"context"
	"fmt"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// createValkeyMemoryStore creates a ValkeyStore backend.
func createValkeyMemoryStore(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, error) {
	return createValkeyMemoryStoreWithContext(context.Background(), cfg, sets...)
}

func createValkeyMemoryStoreWithContext(ctx context.Context, cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, error) {
	vc := cfg.Memory.Valkey
	if vc == nil {
		return nil, fmt.Errorf("memory.valkey configuration is required when backend is 'valkey'")
	}

	host := vc.Host
	if host == "" {
		host = "localhost"
	}
	port := vc.Port
	if port <= 0 {
		port = 6379
	}

	embeddingModel := memory.EmbeddingModelType(detectMemoryEmbeddingModel(cfg))

	embeddingConfig := &memory.EmbeddingConfig{
		Model:     embeddingModel,
		Dimension: vc.Dimension,
	}
	if len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(string(embeddingConfig.Model), 0, 0)
		if err != nil {
			return nil, err
		}
		embeddingConfig.Provider = provider
	}

	dimension, err := memory.StorageDimension(vc.Dimension, *embeddingConfig)
	if err != nil {
		return nil, err
	}
	copied := *vc
	vc = &copied
	vc.Dimension = dimension
	embeddingConfig.Dimension = dimension

	logging.Infof("Memory: connecting to Valkey at %s:%d, embedding=%s", host, port, embeddingConfig.Model)

	clientConfig, err := buildValkeyClientConfig(vc, host, port, remainingContextTimeout(ctx))
	if err != nil {
		return nil, err
	}

	valkeyClient, err := glide.NewClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Valkey client: %w", err)
	}

	store, err := memory.NewValkeyStoreWithContext(ctx, memory.ValkeyStoreOptions{
		Client:             valkeyClient,
		Config:             cfg.Memory,
		ValkeyConfig:       vc,
		Enabled:            true,
		EmbeddingConfig:    embeddingConfig,
		CloseClientOnClose: true,
	})
	if err != nil {
		valkeyClient.Close()
		return nil, fmt.Errorf("failed to create Valkey memory store: %w", err)
	}

	logging.Infof("Memory store initialized: backend=valkey, address=%s:%d, embedding=%s",
		host, port, embeddingConfig.Model)

	return store, nil
}

// buildValkeyClientConfig constructs the valkey-glide client configuration.
func buildValkeyClientConfig(vc *config.MemoryValkeyConfig, host string, port int, attemptTimeout ...time.Duration) (*glideconfig.ClientConfiguration, error) {
	clientConfig := glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{
			Host: host,
			Port: port,
		}).
		WithClientName("vllm_agentic_memory_client")

	if vc.Password != "" {
		clientConfig = clientConfig.WithCredentials(
			glideconfig.NewServerCredentials("", vc.Password),
		)
	}

	if vc.Database != 0 {
		clientConfig = clientConfig.WithDatabaseId(vc.Database)
	}

	requestTimeout := time.Duration(vc.Timeout) * time.Second
	var connectTimeout time.Duration
	if len(attemptTimeout) > 0 && attemptTimeout[0] > 0 {
		connectTimeout = attemptTimeout[0]
		if requestTimeout <= 0 || connectTimeout < requestTimeout {
			requestTimeout = connectTimeout
		}
	}
	if requestTimeout > 0 {
		clientConfig = clientConfig.WithRequestTimeout(requestTimeout)
	}

	advanced := glideconfig.NewAdvancedClientConfiguration()
	useAdvanced := false
	if connectTimeout > 0 {
		advanced = advanced.WithConnectionTimeout(connectTimeout)
		useAdvanced = true
	}
	if vc.TLSEnabled {
		tlsCfg, tlsErr := buildValkeyTLSConfig(vc)
		if tlsErr != nil {
			return nil, tlsErr
		}
		advanced = advanced.WithTlsConfiguration(tlsCfg)
		useAdvanced = true
		clientConfig = clientConfig.WithUseTLS(true)
		logging.Infof("Memory: Valkey TLS enabled (ca_path=%q, insecure_skip_verify=%v)", vc.TLSCAPath, vc.TLSInsecureSkipVerify)
	}
	if useAdvanced {
		clientConfig = clientConfig.WithAdvancedConfiguration(advanced)
	}

	return clientConfig, nil
}

func remainingContextTimeout(ctx context.Context) time.Duration {
	if ctx == nil {
		return 0
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		return 0
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return time.Nanosecond
	}
	return remaining
}

// buildValkeyTLSConfig constructs a glide TLS configuration from the Valkey config.
func buildValkeyTLSConfig(vc *config.MemoryValkeyConfig) (*glideconfig.TlsConfiguration, error) {
	tlsConfig := glideconfig.NewTlsConfiguration()
	if vc.TLSCAPath != "" {
		caCert, err := glideconfig.LoadRootCertificatesFromFile(vc.TLSCAPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS CA certificate from %s: %w", vc.TLSCAPath, err)
		}
		tlsConfig = tlsConfig.WithRootCertificates(caCert)
	}
	if vc.TLSInsecureSkipVerify {
		tlsConfig = tlsConfig.WithInsecureTLS(true)
		logging.Warnf("Memory: Valkey TLS certificate verification is DISABLED — do not use in production")
	}
	return tlsConfig, nil
}
