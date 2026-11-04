package extproc

import (
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	requestDemandStageOriginal       = "original"
	requestDemandStagePostContext    = "post_context"
	requestDemandStagePostToolPolicy = "post_tool_policy"
	requestDemandStageProviderBound  = "provider_bound"

	requestDemandCountingSourceFallback = "fallback"
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

// captureRequestDemand records a bounded, content-free estimate of the neutral
// request at one lifecycle stage. The representation field makes explicit that
// provider_bound describes the final semantic request before wire encoding; the
// estimate deliberately uses the existing provider-neutral fallback until #3050
// supplies tokenizer-aware accounting. This evidence is observe-only: it never
// changes routing or admission.
func captureRequestDemand(
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

	snapshot, ok := reusableRequestDemandSnapshot(ctx.RequestDemandSnapshots, request.Generation)
	if ok {
		snapshot.Stage = stage
		snapshot.Model = firstNonEmpty(strings.TrimSpace(model), strings.TrimSpace(request.Model))
	} else {
		facts := extractSemanticRequestSignals(request)
		snapshot = requestDemandSnapshot(stage, request, model, facts.ContextTokenFloor)
	}
	recordRequestDemandSnapshot(ctx, snapshot)
}

func captureOriginalRequestDemand(
	ctx *RequestContext,
	request *llmprotocol.Request,
	facts *requestSignalSnapshot,
) {
	if ctx == nil || request == nil || facts == nil {
		return
	}
	recordRequestDemandSnapshot(ctx, requestDemandSnapshot(
		requestDemandStageOriginal,
		request,
		request.Model,
		facts.ContextTokenFloor,
	))
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
	promptTokens int,
) routerreplay.RequestDemandSnapshot {
	if promptTokens < 0 {
		promptTokens = 0
	}
	reservedOutput, reserveSource := requestOutputReserve(request)
	return routerreplay.RequestDemandSnapshot{
		Stage:                stage,
		Representation:       requestDemandRepresentationSemantic,
		Model:                firstNonEmpty(strings.TrimSpace(model), strings.TrimSpace(request.Model)),
		PromptTokens:         promptTokens,
		ReservedOutputTokens: reservedOutput,
		TotalDemandTokens:    saturatingNeutralAdd(promptTokens, reservedOutput),
		TotalDemandKnown:     reserveSource == requestDemandOutputReserveExplicit,
		CountingSource:       requestDemandCountingSourceFallback,
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

func reusableRequestDemandSnapshot(
	snapshots []routerreplay.RequestDemandSnapshot,
	generation uint64,
) (routerreplay.RequestDemandSnapshot, bool) {
	if len(snapshots) == 0 {
		return routerreplay.RequestDemandSnapshot{}, false
	}
	latest := snapshots[len(snapshots)-1]
	return latest, latest.RequestGeneration == generation
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
