package extproc

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

const maxBenchmarkUsageReceiptBytes = 12 * 1024

var errBenchmarkCallLimit = errors.New("the selected route cannot enforce the requested inference call limit")

// Only these transformations are known not to dispatch auxiliary generative
// calls. Unknown plugins and future algorithms fail closed for cost proof.
func (r *OpenAIRouter) benchmarkUsageScopeKnown(ctx *RequestContext, allowLooper bool) bool {
	if r == nil || r.Config == nil || ctx == nil || ctx.SkipProcessing || r.Config.Memory.Enabled || ctx.ShadowDispatchPluginConfig != nil {
		return false
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil {
		return !r.Config.ModelSelection.Enabled
	}
	method := "static"
	if decision.Algorithm != nil {
		method = strings.ToLower(strings.TrimSpace(decision.Algorithm.Type))
	}
	if method == "" {
		method = "static"
	}
	if config.IsLooperAlgorithmType(method) {
		if !allowLooper {
			return false
		}
	} else {
		switch method {
		case "static", "automix", "hybrid", "router_dc", "knn", "kmeans", "svm", "mlp", "latency_aware", "multi_factor":
		default:
			return false
		}
	}
	if r.Config.ModelSelection.Enabled && strings.EqualFold(r.Config.ModelSelection.Method, "prompt") {
		return false
	}
	for _, plugin := range decision.Plugins {
		switch config.NormalizeDecisionPluginType(plugin.Type) {
		case config.DecisionPluginSystemPrompt, config.DecisionPluginHeaderMutation, config.DecisionPluginRequestParams, config.DecisionPluginRouterReplay, config.DecisionPluginResponseCache:
		case config.DecisionPluginContextCompression:
			compression := decision.GetContextCompressionConfig()
			if compression == nil || compression.Enabled && compression.EffectiveScoring().Method != config.ContextCompressionScoringBM25 {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func (r *OpenAIRouter) benchmarkCallLimitCheck(ctx *RequestContext) error {
	if headerValueCI(ctx, headers.SRBenchMaxInferenceCalls) != "" && !r.benchmarkUsageScopeKnown(ctx, false) {
		return errBenchmarkCallLimit
	}
	return nil
}

type benchmarkUsageCall struct {
	Model  string                `json:"model"`
	Role   string                `json:"role"`
	Stage  string                `json:"stage"`
	Status string                `json:"status"`
	Usage  benchmarkUsageBuckets `json:"usage"`
}
type benchmarkUsageBuckets struct {
	Prompt     int64 `json:"prompt_tokens"`
	Completion int64 `json:"completion_tokens"`
	Total      int64 `json:"total_tokens"`
	Cached     int64 `json:"cached_input_tokens"`
	Write      int64 `json:"cache_write_tokens"`
}
type benchmarkUsageReceipt struct {
	Version  int                  `json:"version"`
	Complete bool                 `json:"complete"`
	Calls    []benchmarkUsageCall `json:"calls"`
}

func (r *OpenAIRouter) benchmarkLooperUsage(resp *looper.Response, ctx *RequestContext) string {
	receipt := benchmarkUsageReceipt{Version: 1, Calls: []benchmarkUsageCall{}}
	if resp != nil && resp.ExecutionTrace.Version == looper.ExecutionTraceVersion {
		trace := resp.ExecutionTrace
		receipt.Complete = r.benchmarkUsageScopeKnown(ctx, true) && !trace.AttemptsTruncated && trace.DroppedAttemptCount == 0 && len(trace.Attempts) > 0
		var total looper.TokenUsage
		for _, attempt := range trace.Attempts {
			u := attempt.Usage
			if attempt.Model == "" || attempt.Status != looper.AttemptStatusSucceeded || !u.Complete() {
				receipt.Complete = false
			}
			receipt.Calls = append(receipt.Calls, benchmarkUsageCall{Model: attempt.Model, Role: attempt.Role, Stage: attempt.Stage, Status: string(attempt.Status), Usage: benchmarkUsageBuckets{u.PromptTokens, u.CompletionTokens, u.TotalTokens, u.CachedInputTokens, u.CacheWriteTokens}})
			total = total.Add(&looper.ModelResponse{Usage: u})
		}
		if !total.Complete() || total.PromptTokens != resp.Usage.PromptTokens || total.CompletionTokens != resp.Usage.CompletionTokens || total.CachedInputTokens != resp.Usage.CachedInputTokens || total.CacheWriteTokens != resp.Usage.CacheWriteTokens {
			receipt.Complete = false
		}
	}
	encoded, err := json.Marshal(receipt)
	if err != nil || len(encoded) > maxBenchmarkUsageReceiptBytes {
		return `{"version":1,"complete":false,"calls":[]}`
	}
	return string(encoded)
}
