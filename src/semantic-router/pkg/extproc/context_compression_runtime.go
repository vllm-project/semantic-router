package extproc

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
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

// contextCompressionScorer judges the compression targets with the compress
// block's match, the decision model's relevance by default.
func (r *OpenAIRouter) contextCompressionScorer(compress *config.ContextCompressConfig) contextcompression.RelevanceScorer {
	return matchScorer{matcher: r.matcher(), block: compress.Match}
}

// matchScorer scores compression targets by a match block. A target the
// match does not keep scores below every kept one, so it shrinks first.
type matchScorer struct {
	matcher *match.Matcher
	block   *config.MatchConfig
}

func (s matchScorer) Score(ctx context.Context, query string, texts []string) ([]float64, error) {
	scores := make([]float64, len(texts))
	if strings.TrimSpace(query) == "" {
		return scores, nil
	}
	candidates := make([]match.Candidate, len(texts))
	for index, text := range texts {
		candidates[index] = match.Candidate{ID: strconv.Itoa(index), Text: text}
	}
	result, err := s.matcher.Match(ctx, config.MatchCompress, s.block, match.Input{Request: query, Candidates: candidates})
	if err != nil {
		return nil, err
	}
	for index := range scores {
		scores[index] = -1
	}
	for _, kept := range result.Kept {
		if index, convErr := strconv.Atoi(kept.ID); convErr == nil {
			scores[index] = kept.Score
		}
	}
	return scores, nil
}

// contextCompression is the decision's context plugin when it compresses.
func contextCompression(decision *config.Decision) *config.ContextPluginConfig {
	if decision == nil {
		return nil
	}
	plugin := decision.GetContextConfig()
	if plugin == nil || !plugin.Enabled || plugin.Compress == nil {
		return nil
	}
	return plugin
}

func (r *OpenAIRouter) contextCompressionRecoveryStore(
	cfg *config.ContextCompressConfig,
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
