package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// setFactCheckFromSignals records, for memory search and RAG, whether the
// request's fact_check signal said it needs fact checking and whether it
// carries tools or tool results. Signal names: "needs_fact_check" or
// "no_fact_check_needed".
func (r *OpenAIRouter) setFactCheckFromSignals(ctx *RequestContext, matchedFactCheckRules []string) {
	if len(matchedFactCheckRules) == 0 {
		logging.Debugf("No fact-check signals matched")
		return
	}

	needsFactCheck := false
	for _, rule := range matchedFactCheckRules {
		if rule == "needs_fact_check" {
			needsFactCheck = true
			break
		}
	}

	ctx.FactCheckNeeded = needsFactCheck
	// The signal already passed its threshold; its raw confidence is not kept.
	ctx.FactCheckConfidence = 1.0

	logging.Infof("Fact-check from signals: needs_fact_check=%v, matched_rules=%v",
		needsFactCheck, matchedFactCheckRules)

	r.checkRequestHasTools(ctx)
}

// checkRequestHasTools records whether the request carries retrieved context,
// tool definitions or tool results.
func (r *OpenAIRouter) checkRequestHasTools(ctx *RequestContext) {
	if ctx.RAGRetrievedContext != "" {
		ctx.HasToolsForFactCheck = true
		return
	}
	if ctx.SemanticRequest == nil {
		return
	}
	if len(ctx.SemanticRequest.Tools) > 0 || len(extractSemanticToolResults(ctx.SemanticRequest.Messages)) > 0 {
		ctx.HasToolsForFactCheck = true
	}
}

func extractSemanticToolResults(messages []llmprotocol.Message) []string {
	var results []string
	for _, message := range messages {
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
				continue
			}
			if text := semanticText(content.ToolResult.Content); text != "" {
				results = append(results, text)
			}
		}
	}
	return results
}
