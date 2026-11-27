package looper

import (
	"context"

	"github.com/openai/openai-go"
)

// StreamingSynthesisPlan contains the prepared parameters and intermediate
// evidence needed to dispatch a native streaming final stage through Envoy.
type StreamingSynthesisPlan struct {
	SynthesisParams   *openai.ChatCompletionNewParams
	TargetModel       string               // Upstream model target for dispatch
	IntermediateUsage TokenUsage           // Aggregated usage from intermediate stages (panel + analysis)
	Trace             any                  // Intermediate execution evidence for replay (e.g., *FusionTrace, *workflowTrace, []RoundResponse)
	ModelsUsed        []string             // Unique models that contributed to intermediate stages
	Iterations        int                  // Number of intermediate iterations executed
	QuorumOutcome     *FusionQuorumOutcome // Quorum decision evidence (if fallback occurred)
}

// StreamingLooper is optionally implemented by loopers that support preparing
// native streaming execution for eligible final stages.
type StreamingLooper interface {
	Looper
	PrepareStreamingSynthesis(ctx context.Context, req *Request) (*StreamingSynthesisPlan, error)
}
