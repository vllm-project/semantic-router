package dsl

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type typedPluginConfigEmitter func(*strings.Builder, *config.DecisionPlugin)

var typedPluginConfigEmitters = map[string]typedPluginConfigEmitter{
	"system_prompt":   emitSystemPromptPluginConfig,
	"response_cache":  emitResponseCachePluginConfig,
	"context":         emitStructuredPluginConfig,
	"prompt_cache":    emitStructuredPluginConfig,
	"router_replay":   emitRouterReplayPluginConfig,
	"response_guard":  emitStructuredPluginConfig,
	"fast_response":   emitFastResponsePluginConfig,
	"request_params":  emitRequestParamsPluginConfig,
	"tools":           emitStructuredPluginConfig,
	"header_mutation": emitHeaderMutationPluginConfig,
	"masking":         emitStructuredPluginConfig,
}

func decompilePluginConfig(p *config.DecisionPlugin) string {
	var sb strings.Builder
	emitTypedPluginConfig(&sb, p)
	if rawMap, ok := normalizePluginConfigMap(p.Configuration); ok {
		extraFields := filterPluginConfigMap(rawMap, knownPluginConfigKeys(p))
		if len(extraFields) > 0 {
			writePluginConfigMap(&sb, extraFields, "    ")
		}
	}
	return sb.String()
}

func emitTypedPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	if fn, ok := typedPluginConfigEmitters[config.NormalizeDecisionPluginType(p.Type)]; ok {
		fn(sb, p)
	}
}

func emitSystemPromptPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.SystemPromptPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled != nil {
		fmt.Fprintf(sb, "    enabled: %v\n", *cfg.Enabled)
	}
	if cfg.SystemPrompt != "" {
		fmt.Fprintf(sb, "    system_prompt: %q\n", cfg.SystemPrompt)
	}
	if cfg.Mode != "" {
		fmt.Fprintf(sb, "    mode: %q\n", cfg.Mode)
	}
}

func emitResponseCachePluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	emitStructuredPluginConfig(sb, p)
}

func emitStructuredPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	raw, ok := normalizePluginConfigMap(p.Configuration)
	if !ok {
		return
	}
	writePluginConfigMap(sb, raw, "    ")
}

func emitRouterReplayPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	raw, ok := normalizePluginConfigMap(p.Configuration)
	if !ok {
		return
	}
	// Keep the canonical field order while retaining raw omission, false and
	// zero; an effective capture struct would erase the inheritance contract.
	typ := reflect.TypeOf(config.RouterReplayPluginConfig{})
	for index := 0; index < typ.NumField(); index++ {
		key := strings.Split(typ.Field(index).Tag.Get("json"), ",")[0]
		if value, exists := raw[key]; exists {
			fmt.Fprintf(sb, "    %s: %s\n", key, formatPluginConfigValue(value))
			delete(raw, key)
		}
	}
	writePluginConfigMap(sb, raw, "    ")
}

func emitFastResponsePluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.FastResponsePluginConfig](p)
	if !ok {
		return
	}
	if cfg.Message != "" {
		fmt.Fprintf(sb, "    message: %q\n", cfg.Message)
	}
}

func emitRequestParamsPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.RequestParamsPluginConfig](p)
	if !ok {
		return
	}
	if len(cfg.BlockedParams) > 0 {
		fmt.Fprintf(sb, "    blocked_params: %s\n", formatStringArray(cfg.BlockedParams))
	}
	if cfg.DefaultMaxTokens != nil {
		if cfg.DefaultMaxTokens.Auto {
			fmt.Fprintln(sb, "    default_max_tokens: \"auto\"")
		} else {
			fmt.Fprintf(sb, "    default_max_tokens: %d\n", cfg.DefaultMaxTokens.Value)
		}
	}
	if cfg.MaxTokensLimit != nil {
		fmt.Fprintf(sb, "    max_tokens_limit: %d\n", *cfg.MaxTokensLimit)
	}
	if cfg.MaxN != nil {
		fmt.Fprintf(sb, "    max_n: %d\n", *cfg.MaxN)
	}
}

func emitHeaderMutationPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.HeaderMutationPluginConfig](p)
	if !ok {
		return
	}
	if len(cfg.Add) > 0 {
		fmt.Fprintf(sb, "    add: [")
		for i, h := range cfg.Add {
			if i > 0 {
				fmt.Fprintf(sb, ", ")
			}
			fmt.Fprintf(sb, "{ name: %q, value: %q }", h.Name, h.Value)
		}
		fmt.Fprintf(sb, "]\n")
	}
	if len(cfg.Update) > 0 {
		fmt.Fprintf(sb, "    update: [")
		for i, h := range cfg.Update {
			if i > 0 {
				fmt.Fprintf(sb, ", ")
			}
			fmt.Fprintf(sb, "{ name: %q, value: %q }", h.Name, h.Value)
		}
		fmt.Fprintf(sb, "]\n")
	}
	if len(cfg.Delete) > 0 {
		fmt.Fprintf(sb, "    delete: %s\n", formatStringArray(cfg.Delete))
	}
}
