package config

import "fmt"

func ValidateRequestParamsPluginConfig(cfg *RequestParamsPluginConfig) error {
	if cfg != nil && cfg.DefaultMaxTokens != nil {
		if err := cfg.DefaultMaxTokens.Validate(); err != nil {
			return err
		}
		if cfg.DefaultMaxTokens.IsAuto() && cfg.MaxTokensLimit != nil && *cfg.MaxTokensLimit <= 0 {
			return fmt.Errorf("automatic default_max_tokens requires a positive max_tokens_limit when configured")
		}
	}
	return nil
}
