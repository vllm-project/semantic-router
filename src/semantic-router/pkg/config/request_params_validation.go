package config

import (
	"fmt"
	"strings"
)

func ValidateRequestParamsPluginConfig(cfg *RequestParamsPluginConfig) error {
	if cfg == nil {
		return nil
	}
	// The Router streams the answer exactly when the client asks for a stream,
	// so stripping stream would answer a stream reader with one JSON body.
	for _, field := range cfg.BlockedParams {
		if strings.TrimSpace(field) == "stream" {
			return fmt.Errorf("blocked_params cannot include stream: the Router streams the answer when the client asks for a stream")
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
