package contextcompression

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// PolicyFromConfig is the policy of a context plugin's compress block within
// its budget; targetOverride, a request's target directive, replaces the
// budget's target.
func PolicyFromConfig(
	cfg *config.ContextCompressConfig,
	budget *config.ContextBudgetConfig,
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
		targets.CurrentUser = TargetPolicy{Mode: TargetMode(cfg.Targets.CurrentUser.Mode)}
		targets.History = targetPolicyFromConfig(cfg.Targets.History)
		targets.RAG = targetPolicyFromConfig(cfg.Targets.RAG)
		targets.Memory = targetPolicyFromConfig(cfg.Targets.Memory)
	}
	policy := Policy{
		Mode:        Mode(cfg.EffectiveMode()),
		Budget:      budgetFromConfig(budget),
		Targets:     targets,
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

// budgetFromConfig reads the context budget: an explicit max_tokens both
// starts compression and is its target; auto resolves both from the model's
// context window less the reserved output.
func budgetFromConfig(
	budget *config.ContextBudgetConfig,
) Budget {
	if budget == nil {
		return Budget{
			TriggerAuto: true,
			TargetAuto:  true,
			ReserveAuto: true,
		}
	}
	maxTokens := budget.MaxTokens
	return Budget{
		TriggerTokens:       tokenLimitValue(maxTokens),
		TargetTokens:        tokenLimitValue(maxTokens),
		ReserveOutputTokens: tokenLimitValue(budget.ReserveOutputTokens),
		TriggerAuto:         maxTokens == nil || maxTokens.Auto,
		TargetAuto:          maxTokens == nil || maxTokens.Auto,
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
