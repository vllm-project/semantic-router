package extproc

import (
	"encoding/json"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

const messagesCountPath = "/v1/messages/count_tokens"

// countTokensFields are the request fields Anthropic's count_tokens accepts,
// plus vLLM's chat_template_kwargs, which the template renders.
var countTokensFields = []string{"model", "messages", "system", "tools", "tool_choice", "thinking", "chat_template_kwargs"}

func isCountTokensRequest(ctx *RequestContext) bool {
	return ctx != nil && normalizeRequestPath(ctx.Headers[":path"]) == messagesCountPath
}

// withCountTokensOutputLimit gives a count_tokens body the max_tokens the
// Messages decoder requires. The limit is never sent: it stays above any
// thinking budget so the decoder accepts the request as it would a turn.
func withCountTokensOutputLimit(body []byte) []byte {
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || fields == nil || fields["max_tokens"] != nil {
		return body
	}
	var thinking struct {
		BudgetTokens int64 `json:"budget_tokens"`
	}
	_ = json.Unmarshal(fields["thinking"], &thinking)
	fields["max_tokens"], _ = json.Marshal(thinking.BudgetTokens + 1)
	if limited, err := json.Marshal(fields); err == nil {
		return limited
	}
	return body
}

// handleCountTokens answers Anthropic's count_tokens without routing a turn,
// so nothing is recorded in the session, Replay or Router Learning. It counts
// with the Messages backend of the named concrete model, or of the model that
// last served the session when no recipe decision guards content, and
// otherwise answers with the Router's own input estimate.
func (r *OpenAIRouter) handleCountTokens(
	request *llmprotocol.Request,
	model string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	r.resolveEntrypointForRequest(model, ctx)
	if r.requestModelActsAsAuto(model) {
		model, _, _ = sessiontelemetry.PeekLastModelInfo(routingSessionStateKey(ctx), time.Time{})
	} else if unavailable := r.unavailableModelResponse(model, ctx); unavailable != nil {
		return unavailable, nil
	}
	// Candidate requirements can exclude the remembered model for this input.
	if model != "" && !recipeGuardsContent(ctx) && !selection.CandidateRequirementsEnabled(r.candidateRequirements(ctx)) {
		if limited := r.applyRateLimit(ctx, model); limited != nil {
			return limited, nil
		}
		if response, ok, err := r.forwardCountTokens(request, model, ctx); ok || err != nil {
			return response, err
		}
	}
	// The count is already Anthropic's wire shape, not a turn to re-encode.
	ctx.ImmediateResponseEncoded = true
	return r.createJSONResponse(200, map[string]int{"input_tokens": estimateCountTokens(request)}), nil
}

// estimateCountTokens counts text by pre-tokenizer pieces, a whitespace run as
// one token and any other piece as one token per six bytes. On 803 recorded
// agent texts it was within 25% of the Qwen3.8 tokenizer for 95% of them,
// against 39% for the routing estimate's four bytes per token, which also
// counts a long whitespace run as many tokens. Everything else in the request
// keeps the routing estimate.
func estimateCountTokens(request *llmprotocol.Request) int {
	textBytes, textTokens := 0, 0
	var walk func([]llmprotocol.Content)
	walk = func(contents []llmprotocol.Content) {
		for _, content := range contents {
			switch content.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
				textBytes += len(content.Text)
				textTokens += textPieceTokens(content.Text)
			case llmprotocol.ContentToolResult:
				if content.ToolResult != nil {
					walk(content.ToolResult.Content)
				}
			}
		}
	}
	for _, message := range request.Instructions {
		walk(message.Content)
	}
	for _, message := range request.Messages {
		walk(message.Content)
	}
	routing := llmprotocol.EstimateInput(request).Tokens
	return max(0, routing-(textBytes+llmprotocol.InputTextBytesPerToken-1)/llmprotocol.InputTextBytesPerToken) + textTokens
}

// contentFreeSignalTypes read only the shape and size of a conversation.
var contentFreeSignalTypes = map[string]bool{config.SignalTypeConversation: true, config.SignalTypeContext: true}

// recipeGuardsContent reports whether a recipe decision could keep this
// content from the model that last served the session: one that reads content
// to choose a model or refuse, or one with a plugin, since plugins can mask,
// strip or answer in place of the backend. count_tokens runs no decision, so
// such a recipe gets the Router's estimate.
func recipeGuardsContent(ctx *RequestContext) bool {
	recipe := ctx.Routing.SelectedRecipe()
	if recipe == nil {
		return false
	}
	for i := range recipe.Profile.Decisions {
		if len(recipe.Profile.Decisions[i].Plugins) > 0 || ruleReadsContent(recipe.Profile.Decisions[i].Rules) {
			return true
		}
	}
	return false
}

func ruleReadsContent(node config.RuleNode) bool {
	if node.IsLeaf() {
		return !contentFreeSignalTypes[node.Type]
	}
	for _, child := range node.Conditions {
		if ruleReadsContent(child) {
			return true
		}
	}
	return false
}

// forwardCountTokens sends the request to the model's Messages backend at its
// count_tokens path. The response passes back unchanged.
func (r *OpenAIRouter) forwardCountTokens(
	request *llmprotocol.Request,
	model string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, bool, error) {
	dispatch, err := r.resolveProviderDispatchForCandidate(model, "", false, ctx)
	if err != nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 {
		return nil, false, nil
	}
	changed, err := r.prepareProviderRequest(request, dispatch, ctx)
	if err != nil {
		return nil, false, err
	}
	if changed {
		request.Generation++
	}
	ctx.SemanticRequest, ctx.TargetFormat, ctx.RequestModel = request, dispatch.targetFormat, dispatch.logicalModel
	body, err := r.encodeDispatchRequest(ctx)
	if err == nil {
		body, err = r.adaptProviderRequest(body, dispatch, ctx)
	}
	if err == nil {
		body, err = countTokensBody(body)
	}
	if err != nil {
		return nil, false, dispatchWireError(err, ctx, "encode count_tokens request")
	}
	response := r.buildProviderDispatchResponse(dispatch, ctx)
	common := response.GetRequestBody().GetResponse()
	if common == nil {
		return response, true, nil
	}
	for _, header := range common.HeaderMutation.SetHeaders {
		if header.GetHeader().GetKey() == ":path" {
			header.Header.RawValue = append(header.Header.RawValue, "/count_tokens"...)
		}
	}
	appendContentLengthHeader(&common.HeaderMutation.SetHeaders, len(body))
	common.BodyMutation = &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{Body: body}}
	ctx.SkipProcessing = true
	return response, true, nil
}

func countTokensBody(body []byte) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	kept := make(map[string]json.RawMessage, len(countTokensFields))
	for _, name := range countTokensFields {
		if value, ok := fields[name]; ok {
			kept[name] = value
		}
	}
	return json.Marshal(kept)
}

// textPieceTokens splits text as Qwen2's BPE pre-tokenizer pattern does
// without its look-ahead, in one pass:
//
//	'(?i:s|t|re|ve|m|ll|d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//	 ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+
//
// and counts a whitespace piece as one token and any other piece as one per
// six bytes.
func textPieceTokens(text string) int {
	tokens := 0
	for i := 0; i < len(text); {
		end := textPieceEnd(text, i)
		if strings.TrimSpace(text[i:end]) == "" {
			tokens++
		} else {
			tokens += (end - i + 5) / 6
		}
		i = end
	}
	return tokens
}

func textPieceEnd(text string, i int) int {
	r, size := utf8.DecodeRuneInString(text[i:])
	next, _ := utf8.DecodeRuneInString(text[i+size:])
	hasNext := i+size < len(text)
	switch {
	case r == '\'' && contractionLength(text[i+size:]) > 0:
		return i + size + contractionLength(text[i+size:])
	case unicode.IsLetter(r):
		return runEnd(text, i+size, unicode.IsLetter)
	case r != '\r' && r != '\n' && !unicode.IsNumber(r) && hasNext && unicode.IsLetter(next):
		return runEnd(text, i+size, unicode.IsLetter)
	case unicode.IsNumber(r):
		return i + size
	case r == ' ' && hasNext && isOtherRune(next):
		return runEnd(text, runEnd(text, i+size, isOtherRune), isNewline)
	case isOtherRune(r):
		return runEnd(text, runEnd(text, i, isOtherRune), isNewline)
	}
	end := runEnd(text, i, isSpaceRune)
	if last := strings.LastIndexAny(text[i:end], "\r\n"); last >= 0 {
		return i + last + 1
	}
	return end
}

// runEnd returns where the run of runes matching in, starting at i, ends.
func runEnd(text string, i int, in func(rune) bool) int {
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if !in(r) {
			break
		}
		i += size
	}
	return i
}

func contractionLength(rest string) int {
	for _, suffix := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
		if len(rest) >= len(suffix) && strings.EqualFold(rest[:len(suffix)], suffix) {
			return len(suffix)
		}
	}
	return 0
}

// isSpaceRune is RE2's \s.
func isSpaceRune(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\f' || r == '\r'
}

func isNewline(r rune) bool { return r == '\r' || r == '\n' }

func isOtherRune(r rune) bool {
	return !isSpaceRune(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}
