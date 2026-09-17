package config

import "fmt"

func ValidateRequestParamsPluginConfig(cfg *RequestParamsPluginConfig) error {
	if cfg != nil && cfg.DefaultMaxTokens != nil && *cfg.DefaultMaxTokens <= 0 {
		return fmt.Errorf("default_max_tokens must be a positive integer")
	}
	return nil
}
