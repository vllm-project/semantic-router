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

// captureRequestDemand records a bounded, content-free count of the neutral
// request at one lifecycle stage. The representation field makes explicit that
// provider_bound describes the final semantic request before wire encoding.
// The prompt is counted with the tokenizer the context window check uses for
// the stage's model (#3050): exact when the model names one, the proxy
// tokenizer otherwise, and the provider-neutral fallback for media, which no
// text tokenizer counts. This evidence is observe-only: it never changes
// routing or admission.
func (r *OpenAIRouter) captureRequestDemand(
	ctx *RequestContext,
	stage string,
	request *llmprotocol.Request,
	model string,
) {
	r.recordRequestDemand(ctx, stage, request, model, nil)
}

// captureOriginalRequestDemand records the original stage; facts, the
// request's signals, supply the fallback without another walk.
func (r *OpenAIRouter) captureOriginalRequestDemand(
	ctx *RequestContext,
	request *llmprotocol.Request,
	facts *requestSignalSnapshot,
) {
	if facts == nil {
		return
	}
	r.recordRequestDemand(ctx, requestDemandStageOriginal, request, "", facts)
}

func (r *OpenAIRouter) recordRequestDemand(
	ctx *RequestContext,
	stage string,
	request *llmprotocol.Request,
	model string,
	facts *requestSignalSnapshot,
) {
	if ctx == nil || request == nil {
		return
	}
	if _, ok := requestDemandStageOrder[stage]; !ok {
		return
	}
	model = firstNonEmpty(strings.TrimSpace(model), strings.TrimSpace(request.Model))
	promptTokens, source := r.requestDemandTokens(ctx, request, model, facts)
	recordRequestDemandSnapshot(ctx, requestDemandSnapshot(stage, request, model, promptTokens, source))
}

// requestDemandTokens counts the prompt with the stage model's tokenizer. A
// count is made once per tokenizer and request generation: stages that leave
// the request unchanged, and the context window check, share it.
func (r *OpenAIRouter) requestDemandTokens(
	ctx *RequestContext,
	request *llmprotocol.Request,
	model string,
	facts *requestSignalSnapshot,
) (int, string) {
	if r != nil && r.Config != nil && !requestCarriesMedia(request) {
		if tokenizer := r.requestTokenizer(ctx, model); tokenizer.encoding != nil {
			return requestDemandCount(ctx, request, tokenizer), string(tokenizer.kind)
		}
	}
	if latest, ok := reusableRequestDemandSnapshot(ctx.RequestDemandSnapshots, request.Generation); ok &&
		latest.CountingSource == requestDemandCountingSourceFallback {
		return latest.PromptTokens, requestDemandCountingSourceFallback
	}
	if facts == nil {
		facts = extractSemanticRequestSignals(request)
	}
	return facts.ContextTokenFloor, requestDemandCountingSourceFallback
}

// requestDemandCount counts request with tokenizer, keeping the count of the
// request in flight with the window check's counts, under the key that check
// uses for a prompt no decision reshapes.
func requestDemandCount(ctx *RequestContext, request *llmprotocol.Request, tokenizer modelTokenizer) int {
	if request != ctx.SemanticRequest {
		return countPromptTokens(tokenizer.encoding, request)
	}
	key := contextCountKey{tokenizer: tokenizer.encoding.Name(), generation: request.Generation}
	if count, cached := ctx.contextCounts[key]; cached && count.ok {
		return count.tokens
	}
	tokens := countPromptTokens(tokenizer.encoding, request)
	if ctx.contextCounts == nil {
		ctx.contextCounts = make(map[contextCountKey]contextCount)
	}
	ctx.contextCounts[key] = contextCount{tokens: tokens, ok: true}
	return tokens
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
	countingSource string,
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
		CountingSource:       countingSource,
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
