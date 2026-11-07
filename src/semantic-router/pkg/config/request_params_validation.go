package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func ValidateRequestParamsPluginConfig(cfg *RequestParamsPluginConfig) error {
	if cfg == nil {
		return nil
	}
	for _, field := range cfg.BlockedParams {
		if err := llmprotocol.ValidateBlockableRequestField(field); err != nil {
			return err
		}
	}
	if cfg.DefaultMaxTokens != nil {
		if err := cfg.DefaultMaxTokens.Validate(); err != nil {
			return err
		}
		if cfg.DefaultMaxTokens.IsAuto() && cfg.MaxTokensLimit != nil && *cfg.MaxTokensLimit <= 0 {
			return fmt.Errorf("automatic default_max_tokens requires a positive max_tokens_limit when configured")
		}
	}
	// A limit caps the client's value, so a non-positive one would rewrite every
	// request to an invalid max_tokens or n.
	if cfg.MaxTokensLimit != nil && *cfg.MaxTokensLimit <= 0 {
		return fmt.Errorf("max_tokens_limit must be positive, got %d", *cfg.MaxTokensLimit)
	}
	if cfg.MaxN != nil && *cfg.MaxN <= 0 {
		return fmt.Errorf("max_n must be positive, got %d", *cfg.MaxN)
	}
	return nil
}
