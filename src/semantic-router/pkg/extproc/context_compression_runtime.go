package extproc

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) contextCompressionService() *contextcompression.Service {
	if r == nil {
		return nil
	}
	r.contextCompressionMu.Lock()
	defer r.contextCompressionMu.Unlock()
	if r.ContextCompression == nil {
		r.ContextCompression = contextcompression.NewService()
	}
	return r.ContextCompression
}

func (r *OpenAIRouter) contextCompressionTokenCounter(
	ctx *RequestContext,
) contextcompression.TokenCounter {
	return contextcompression.CalibratedTokenCounter{
		Estimate: func(_ string, byteLength int) (int, bool) {
			if r == nil || r.Classifier == nil {
				return max(0, (byteLength+3)/4), false
			}
			return r.Classifier.EstimateTokens(
				contextCompressionTokenCalibrationCategory(ctx),
				byteLength,
			)
		},
	}
}

func (r *OpenAIRouter) contextCompressionScorer(
	ctx context.Context,
	cfg *config.ContextCompressionPluginConfig,
) contextcompression.RelevanceScorer {
	if cfg == nil || cfg.EffectiveScoring().Method == config.ContextCompressionScoringBM25 {
		return nil
	}
	provider, err := r.contextCompressionEmbeddingProvider()
	if err != nil || provider == nil {
		return nil
	}
	r.contextCompressionMu.Lock()
	defer r.contextCompressionMu.Unlock()
	if r.CompressionScorer == nil {
		r.CompressionScorer = contextcompression.NewMemoizedEmbeddingScorer(
			func(callCtx context.Context, texts []string) ([][]float32, error) {
				return provider.EmbedBatch(callCtx, texts)
			},
			1024,
		)
	}
	return r.CompressionScorer
}

func (r *OpenAIRouter) contextCompressionEmbeddingProvider() (embedding.Provider, error) {
	if r == nil || r.Config == nil {
		return nil, fmt.Errorf("embedding configuration is unavailable")
	}
	r.contextCompressionMu.Lock()
	defer r.contextCompressionMu.Unlock()
	if r.CompressionEmbedding != nil {
		return r.CompressionEmbedding, nil
	}
	provider, err := embedding.NewProviderFromRouterConfig(
		r.Config,
		embedding.ProviderOptions{},
	)
	if err != nil {
		return nil, err
	}
	r.CompressionEmbedding = provider
	return provider, nil
}

func (r *OpenAIRouter) contextCompressionRecoveryStore(
	cfg *config.ContextCompressionPluginConfig,
) contextcompression.RecoveryStore {
	if cfg == nil || cfg.Recovery == nil || !cfg.Recovery.Enabled {
		return nil
	}
	r.contextCompressionMu.Lock()
	defer r.contextCompressionMu.Unlock()
	if r.CompressionRecovery != nil {
		return r.CompressionRecovery
	}
	if r.Config == nil {
		return nil
	}
	storeType := strings.TrimSpace(cfg.Recovery.Store)
	if storeType == "response_cache" {
		storeType = strings.TrimSpace(r.Config.SemanticCache.BackendType)
	}
	options, ok, err := recoveryRedisOptions(
		r.Config,
		storeType,
		cfg.Recovery.MaxTotalBytes,
	)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "context_recovery_store_config_failed", map[string]interface{}{
			"store": storeType,
			"error": err.Error(),
		})
		return nil
	}
	if !ok {
		return nil
	}
	store, err := contextcompression.NewRedisRecoveryStore(options)
	if err != nil {
		return nil
	}
	r.CompressionRecovery = store
	if r.ContextCompression == nil {
		r.ContextCompression = contextcompression.NewService()
	}
	if r.RuntimeRegistry != nil {
		r.RuntimeRegistry.SetContextCompression(
			r.ContextCompression,
			store,
		)
	}
	return store
}

func recoveryRedisOptions(
	routerConfig *config.RouterConfig,
	storeType string,
	maxTotalBytes int,
) (contextcompression.RedisRecoveryStoreOptions, bool, error) {
	switch storeType {
	case "redis":
		if routerConfig.SemanticCache.Redis == nil {
			return contextcompression.RedisRecoveryStoreOptions{}, false, nil
		}
		connection := routerConfig.SemanticCache.Redis.Connection
		tlsConfig, err := recoveryTLSConfig(
			connection.Host,
			connection.TLS.Enabled,
			connection.TLS.CertFile,
			connection.TLS.KeyFile,
			connection.TLS.CAFile,
		)
		if err != nil {
			return contextcompression.RedisRecoveryStoreOptions{}, false, err
		}
		return contextcompression.RedisRecoveryStoreOptions{
			Address:       net.JoinHostPort(connection.Host, strconv.Itoa(connection.Port)),
			Password:      connection.Password,
			Database:      connection.Database,
			TLSConfig:     tlsConfig,
			KeyPrefix:     "vsr:context-recovery:v1:",
			MaxTotalBytes: maxTotalBytes,
		}, true, nil
	case "valkey":
		if routerConfig.SemanticCache.Valkey == nil {
			return contextcompression.RedisRecoveryStoreOptions{}, false, nil
		}
		connection := routerConfig.SemanticCache.Valkey.Connection
		tlsConfig, err := recoveryTLSConfig(
			connection.Host,
			connection.TLS.Enabled,
			connection.TLS.CertFile,
			connection.TLS.KeyFile,
			connection.TLS.CAFile,
		)
		if err != nil {
			return contextcompression.RedisRecoveryStoreOptions{}, false, err
		}
		return contextcompression.RedisRecoveryStoreOptions{
			Address:       net.JoinHostPort(connection.Host, strconv.Itoa(connection.Port)),
			Password:      connection.Password,
			Database:      connection.Database,
			TLSConfig:     tlsConfig,
			KeyPrefix:     "vsr:context-recovery:v1:",
			MaxTotalBytes: maxTotalBytes,
		}, true, nil
	default:
		return contextcompression.RedisRecoveryStoreOptions{}, false, nil
	}
}

func recoveryTLSConfig(
	host string,
	enabled bool,
	certFile string,
	keyFile string,
	caFile string,
) (*tls.Config, error) {
	if !enabled {
		return nil, nil
	}
	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS12,
		ServerName: host,
	}
	if (certFile == "") != (keyFile == "") {
		return nil, fmt.Errorf("recovery TLS cert_file and key_file must be configured together")
	}
	if certFile != "" {
		certificate, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return nil, fmt.Errorf("load recovery TLS client certificate: %w", err)
		}
		tlsConfig.Certificates = []tls.Certificate{certificate}
	}
	if caFile != "" {
		caPEM, err := os.ReadFile(caFile)
		if err != nil {
			return nil, fmt.Errorf("read recovery TLS CA: %w", err)
		}
		roots, err := x509.SystemCertPool()
		if err != nil {
			roots = x509.NewCertPool()
		}
		if !roots.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("recovery TLS CA file contains no certificates")
		}
		tlsConfig.RootCAs = roots
	}
	return tlsConfig, nil
}

func contextCompressionPolicy(
	cfg *config.ContextCompressionPluginConfig,
	targetOverride *int,
) contextcompression.Policy {
	return contextcompression.PolicyFromConfig(cfg, targetOverride)
}

func compressionPolicyForRequest(
	policy contextcompression.Policy,
	ctx *RequestContext,
) contextcompression.Policy {
	if ctx == nil || !ctx.ExpectStreamingResponse {
		return policy
	}
	if policy.Targets.ToolOutputs.Mode == contextcompression.TargetRecoverable {
		policy.Targets.ToolOutputs.Mode = contextcompression.TargetPreserve
	}
	if policy.Targets.History.Mode == contextcompression.TargetRecoverable {
		policy.Targets.History.Mode = contextcompression.TargetPreserve
	}
	if policy.Targets.RAG.Mode == contextcompression.TargetRecoverable {
		policy.Targets.RAG.Mode = contextcompression.TargetPreserve
	}
	if policy.Targets.Memory.Mode == contextcompression.TargetRecoverable {
		policy.Targets.Memory.Mode = contextcompression.TargetPreserve
	}
	policy.Recovery.Enabled = false
	return policy
}

func contextCompressionCapabilities(
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) contextcompression.ModelContextCapabilities {
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	contextWindow := 0
	if routerConfig != nil {
		if params, ok := routerConfig.ModelConfig[model]; ok {
			contextWindow = params.ContextWindowSize
		}
	}
	requestedOutput := requestOutputTokens(ctx.workingRequestBody())
	return contextcompression.ModelContextCapabilities{
		ContextWindow:   contextWindow,
		RequestedOutput: requestedOutput,
	}
}

func requestOutputTokens(body []byte) int {
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		return 0
	}
	for _, field := range []string{"max_completion_tokens", "max_tokens"} {
		switch value := request[field].(type) {
		case float64:
			return int(value)
		case json.Number:
			parsed, _ := strconv.Atoi(string(value))
			return parsed
		}
	}
	return 0
}

func (r *OpenAIRouter) contextCompressionScope(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	userHeader := headers.AuthzUserID
	if r != nil && r.Config != nil {
		userHeader = r.Config.Authz.Identity.GetUserIDHeader()
	}
	user := headerValueCI(ctx, userHeader)
	namespace := cache.UserScopeNamespace(user)
	if namespace == "" || ctx.RequestID == "" {
		return ""
	}
	return cache.CombineFingerprints(
		string(ctx.Routing.RecipeName()),
		ctx.VSRSelectedDecisionName,
		namespace,
		ctx.RequestID,
	)
}
