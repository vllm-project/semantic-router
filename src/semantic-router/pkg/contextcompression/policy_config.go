package contextcompression

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func PolicyFromConfig(
	cfg *config.ContextCompressionPluginConfig,
	targetOverride *int,
) Policy {
	if cfg == nil {
		return Policy{}
	}
	tool := cfg.EffectiveToolOutputTarget()
	targets := Targets{
		ToolOutputs: targetPolicyFromConfig(tool),
		History:     TargetPolicy{Mode: TargetPreserve},
		RAG:         TargetPolicy{Mode: TargetPreserve},
		Memory:      TargetPolicy{Mode: TargetPreserve},
	}
	if cfg.Targets != nil {
		targets.History = targetPolicyFromConfig(cfg.Targets.History)
		targets.RAG = targetPolicyFromConfig(cfg.Targets.RAG)
		targets.Memory = targetPolicyFromConfig(cfg.Targets.Memory)
	}
	policy := Policy{
		Mode:    Mode(cfg.EffectiveMode()),
		Budget:  budgetFromConfig(cfg.Budget),
		Targets: targets,
		Scoring: Scoring{
			Method:            cfg.EffectiveScoring().Method,
			EmbeddingModelRef: cfg.EffectiveScoring().EmbeddingModelRef,
		},
		FailureMode: FailureMode(cfg.EffectiveFailureMode()),
	}
	if targetOverride != nil && *targetOverride > 0 {
		policy.Budget.TargetTokens = *targetOverride
		policy.Budget.TargetAuto = false
	}
	if cfg.Recovery != nil {
		policy.Recovery = RecoveryPolicy{
			Enabled:            cfg.Recovery.Enabled,
			TTL:                durationOrDefault(cfg.Recovery.TTLSeconds, 15*time.Minute),
			MaxBytesPerRequest: intOrDefault(cfg.Recovery.MaxBytesPerRequest, 10*1024*1024),
			MaxTotalBytes:      intOrDefault(cfg.Recovery.MaxTotalBytes, 256*1024*1024),
			MaxRetrievals:      intOrDefault(cfg.Recovery.MaxRetrievals, 8),
		}
	}
	return policy
}

func targetPolicyFromConfig(
	target config.ContextCompressionTargetConfig,
) TargetPolicy {
	mode := strings.TrimSpace(target.Mode)
	if mode == "" {
		mode = config.ContextCompressionTargetPreserve
	}
	return TargetPolicy{
		Mode:         TargetMode(mode),
		MinTokens:    target.MinTokens,
		TargetTokens: target.TargetTokens,
	}
}

func budgetFromConfig(
	budget *config.ContextCompressionBudgetConfig,
) Budget {
	if budget == nil {
		return Budget{
			TriggerAuto: true,
			TargetAuto:  true,
			ReserveAuto: true,
		}
	}
	return Budget{
		TriggerTokens:       tokenLimitValue(budget.TriggerTokens),
		TargetTokens:        tokenLimitValue(budget.TargetTokens),
		ReserveOutputTokens: tokenLimitValue(budget.ReserveOutputTokens),
		TriggerAuto:         budget.TriggerTokens == nil || budget.TriggerTokens.Auto,
		TargetAuto:          budget.TargetTokens == nil || budget.TargetTokens.Auto,
		ReserveAuto:         budget.ReserveOutputTokens == nil || budget.ReserveOutputTokens.Auto,
	}
}

func tokenLimitValue(limit *config.CompressionTokenLimit) int {
	if limit == nil || limit.Auto {
		return 0
	}
	return limit.Value
}

func durationOrDefault(seconds int, fallback time.Duration) time.Duration {
	if seconds <= 0 {
		return fallback
	}
	return time.Duration(seconds) * time.Second
}

func intOrDefault(value int, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}
