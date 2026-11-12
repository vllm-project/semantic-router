package extproc

import (
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokenaccounting"
)

const (
	requestDemandStageOriginal       = "original"
	requestDemandStagePostContext    = "post_context"
	requestDemandStagePostToolPolicy = "post_tool_policy"
	requestDemandStageProviderBound  = "provider_bound"

	requestDemandCountingSourceFallback = string(tokenaccounting.SourceFallback)
	requestDemandRepresentationSemantic = "semantic"
	requestDemandOutputReserveExplicit  = "explicit"
	requestDemandOutputReserveUnknown   = "unknown"

	maxRequestDemandSnapshots = 4
)

var requestDemandStageOrder = map[string]int{
	requestDemandStageOriginal:       0,
	requestDemandStagePostContext:    1,
	requestDemandStagePostToolPolicy: 2,
	requestDemandStageProviderBound:  3,
}

// captureRequestDemand records a bounded, content-free count of the neutral
// request at one lifecycle stage. The representation field makes explicit that
// provider_bound describes the final semantic request before wire encoding.
// This evidence is observe-only: it never changes routing or admission.
func (r *OpenAIRouter) captureRequestDemand(
	ctx *RequestContext,
	stage string,
	request *llmprotocol.Request,
	model string,
) {
	if ctx == nil || request == nil {
		return
	}
	if _, ok := requestDemandStageOrder[stage]; !ok {
		return
	}
	model = firstNonEmpty(strings.TrimSpace(model), strings.TrimSpace(request.Model))
	recordRequestDemandSnapshot(ctx, requestDemandSnapshot(stage, request, model, r.countRequestDemand(ctx, model, request)))
}

// countRequestDemand counts the prompt for the model that receives it:
// exactly with a registered tokenizer, else with the provider-neutral
// estimate, its text rate calibrated when the request's classifier has enough
// provider samples (#3050).
func (r *OpenAIRouter) countRequestDemand(
	ctx *RequestContext,
	model string,
	request *llmprotocol.Request,
) tokenaccounting.Count {
	var counter *tokenaccounting.Counter
	var calibration tokenaccounting.Calibration
	if r != nil {
		counter = r.TokenAccounting
		if classifier := r.classifierForRequest(ctx); classifier != nil {
			// The router-wide category learns from every text-only response,
			// in the text-byte units this count passes.
			calibration = func(textBytes int) (int, bool) {
				return classifier.EstimateTokens("", textBytes)
			}
		}
	}
	return counter.Count(model, request, calibration)
}

func recordRequestDemandSnapshot(
	ctx *RequestContext,
	snapshot routerreplay.RequestDemandSnapshot,
) {
	ctx.RequestDemandSnapshots = upsertRequestDemandSnapshot(ctx.RequestDemandSnapshots, snapshot)
	metrics.RecordRequestDemand(
		snapshot.Stage,
		snapshot.CountingSource,
		snapshot.PromptTokens,
		snapshot.ReservedOutputTokens,
		snapshot.TotalDemandTokens,
		snapshot.TotalDemandKnown,
	)
}

func requestDemandSnapshot(
	stage string,
	request *llmprotocol.Request,
	model string,
	prompt tokenaccounting.Count,
) routerreplay.RequestDemandSnapshot {
	promptTokens := max(prompt.Tokens, 0)
	reservedOutput, reserveSource := requestOutputReserve(request)
	return routerreplay.RequestDemandSnapshot{
		Stage:                stage,
		Representation:       requestDemandRepresentationSemantic,
		Model:                firstNonEmpty(strings.TrimSpace(model), strings.TrimSpace(request.Model)),
		PromptTokens:         promptTokens,
		ReservedOutputTokens: reservedOutput,
		TotalDemandTokens:    saturatingNeutralAdd(promptTokens, reservedOutput),
		TotalDemandKnown:     reserveSource == requestDemandOutputReserveExplicit,
		CountingSource:       string(prompt.Source),
		OutputReserveSource:  reserveSource,
		RequestGeneration:    request.Generation,
	}
}

func requestOutputReserve(request *llmprotocol.Request) (int, string) {
	if request == nil || request.Sampling.MaxOutputTokens == nil {
		return 0, requestDemandOutputReserveUnknown
	}
	value := *request.Sampling.MaxOutputTokens
	if value < 0 {
		return 0, requestDemandOutputReserveUnknown
	}
	if value > int64(math.MaxInt) {
		return math.MaxInt, requestDemandOutputReserveExplicit
	}
	return int(value), requestDemandOutputReserveExplicit
}

func upsertRequestDemandSnapshot(
	snapshots []routerreplay.RequestDemandSnapshot,
	snapshot routerreplay.RequestDemandSnapshot,
) []routerreplay.RequestDemandSnapshot {
	stageIndex, ok := requestDemandStageOrder[snapshot.Stage]
	if !ok {
		return snapshots
	}
	for index := range snapshots {
		if snapshots[index].Stage == snapshot.Stage {
			snapshots[index] = snapshot
			return snapshots
		}
	}
	if len(snapshots) >= maxRequestDemandSnapshots {
		return snapshots
	}

	insertAt := len(snapshots)
	for index := range snapshots {
		if existingIndex, exists := requestDemandStageOrder[snapshots[index].Stage]; exists && existingIndex > stageIndex {
			insertAt = index
			break
		}
	}
	snapshots = append(snapshots, routerreplay.RequestDemandSnapshot{})
	copy(snapshots[insertAt+1:], snapshots[insertAt:])
	snapshots[insertAt] = snapshot
	return snapshots
}

func cloneRequestDemandSnapshots(
	values []routerreplay.RequestDemandSnapshot,
) []routerreplay.RequestDemandSnapshot {
	return append([]routerreplay.RequestDemandSnapshot(nil), values...)
}
