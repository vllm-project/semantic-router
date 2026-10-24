package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type chatResponseWire struct {
	ID      string `json:"id"`
	Object  string `json:"object,omitempty"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	// OpenRouter names the upstream that served the request. It is metadata,
	// not part of the model's answer.
	Provider            *string                   `json:"provider,omitempty"`
	Choices             []chatChoiceWire          `json:"choices"`
	Usage               *chatUsageWire            `json:"usage,omitempty"`
	Metadata            map[string]string         `json:"metadata,omitempty"`
	Moderation          json.RawMessage           `json:"moderation,omitempty"`
	Error               *chatErrorWire            `json:"error,omitempty"`
	ServiceTier         *chatServiceTierWire      `json:"service_tier,omitempty"`
	SystemFingerprint   *string                   `json:"system_fingerprint,omitempty"`
	PromptLogprobs      json.RawMessage           `json:"prompt_logprobs,omitempty"`
	PromptTokenIDs      []int64                   `json:"prompt_token_ids,omitempty"`
	PromptText          json.RawMessage           `json:"prompt_text,omitempty"`
	PromptRoutedExperts json.RawMessage           `json:"prompt_routed_experts,omitempty"`
	KVTransferParams    *chatKVTransferParamsWire `json:"kv_transfer_params,omitempty"`
	ECTransferParams    json.RawMessage           `json:"ec_transfer_params,omitempty"`
	Metrics             json.RawMessage           `json:"metrics,omitempty"`
	DoRemoteDecode      *bool                     `json:"do_remote_decode,omitempty"`
	DoRemotePrefill     *bool                     `json:"do_remote_prefill,omitempty"`
	RemoteBlockIDs      []int64                   `json:"remote_block_ids,omitempty"`
	RemoteEngineID      *string                   `json:"remote_engine_id,omitempty"`
	RemoteHost          *string                   `json:"remote_host,omitempty"`
	RemotePort          *int64                    `json:"remote_port,omitempty"`
	// Groq attaches its request id here; it is provider metadata, not output.
	XGroq json.RawMessage `json:"x_groq,omitempty"`
	// Groq reports per-model usage for compound requests and null otherwise.
	UsageBreakdown json.RawMessage `json:"usage_breakdown,omitempty"`
}

// hasUsageBreakdown reports Groq's per-model usage, which has no neutral slot.
func (wire chatResponseWire) hasUsageBreakdown() bool {
	return len(wire.UsageBreakdown) > 0 && !bytes.Equal(bytes.TrimSpace(wire.UsageBreakdown), []byte("null"))
}

// hasLegacyKVTransferMetadata recognizes the flat KV-transfer response
// metadata emitted by older vLLM-compatible servers. Newer servers place the
// handshake under kv_transfer_params, but both shapes remain in deployed
// provider fleets. The fields stay typed so strict decoding remains intact.
func (wire chatResponseWire) hasLegacyKVTransferMetadata() bool {
	return wire.DoRemoteDecode != nil || wire.DoRemotePrefill != nil || wire.RemoteBlockIDs != nil ||
		wire.RemoteEngineID != nil || wire.RemoteHost != nil || wire.RemotePort != nil
}

type chatChoiceWire struct {
	Index              int               `json:"index"`
	Message            chatMessageWire   `json:"message"`
	FinishReason       *string           `json:"finish_reason"`
	NativeFinishReason *string           `json:"native_finish_reason,omitempty"`
	Logprobs           *chatLogprobsWire `json:"logprobs,omitempty"`
	// LiteLLM wraps upstream fields that do not fit the OpenAI schema in
	// provider_specific_fields on each choice. The content is opaque provider
	// metadata and is not replayed to the client.
	ProviderSpecificFields json.RawMessage     `json:"provider_specific_fields,omitempty"`
	StopReason             *chatStopReasonWire `json:"stop_reason,omitempty"`
	TokenIDs               []int64             `json:"token_ids,omitempty"`
	RoutedExperts          json.RawMessage     `json:"routed_experts,omitempty"`
}

type chatServiceTierWire string

func (tier *chatServiceTierWire) UnmarshalJSON(raw []byte) error {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	switch value {
	// on_demand and performance are Groq's tiers on its OpenAI-compatible API.
	case "auto", "default", "flex", "priority", "scale", "on_demand", "performance":
		*tier = chatServiceTierWire(value)
		return nil
	default:
		return fmt.Errorf("unsupported chat service tier")
	}
}

// chatExecutionFieldNames lists provider execution fields the strict contract
// observes only as null (a non-null body carries data this protocol version
// has not modeled). They decode as raw JSON so the preserve-same-format
// policy can forward them; the strict policy still rejects non-null values
// in validateChatExecutionFields.
var chatExecutionFieldNames = map[string]struct{}{
	"prompt_logprobs": {}, "prompt_text": {}, "prompt_routed_experts": {},
	"ec_transfer_params": {}, "metrics": {}, "routed_experts": {},
}

func isNullJSON(raw json.RawMessage) bool {
	return len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// validateChatExecutionFields enforces the closed provider contract: execution
// fields this protocol only observes as null must stay null. Under the
// preserve-same-format policy the raw values are forwarded instead, so the
// check is skipped entirely.
func validateChatExecutionFields(policy llmprotocol.Policy, fields map[string]json.RawMessage) error {
	if policy.UnknownFields != llmprotocol.UnknownReject {
		return nil
	}
	for name, raw := range fields {
		if _, execution := chatExecutionFieldNames[name]; execution {
			if err := validateChatExecutionField(name, raw, policy); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateChatExecutionField(name string, raw json.RawMessage, policy llmprotocol.Policy) error {
	if policy.UnknownFields != llmprotocol.UnknownReject || isNullJSON(raw) {
		return nil
	}
	// Match the strict decode-time rejection this replaces (the former
	// chatNullOnlyWire unmarshal error): same code, same message.
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"invalid_upstream_json",
		"upstream response JSON is invalid",
		fmt.Errorf("chat execution field %s must be null", name),
	)
}

// chatResponseWireFields is the set of top-level names modelled by
// chatResponseWire. Fields outside it are provider extensions; execution
// fields are modelled only so the strict decoder can reject non-null values.
var chatResponseWireFields = exactJSONStructFields(reflect.TypeOf(chatResponseWire{}))

// captureChatResponsePassthrough returns the top-level fields a same-format
// client must still receive after a re-render: fields outside the modelled
// response wire set plus non-null execution fields. Under the strict policy
// it instead rejects non-null execution fields, keeping the closed provider
// contract unchanged. Preserved bytes are bounded by SourceEnvelopeBytes
// like the request capture.
func captureChatResponsePassthrough(body []byte, policy llmprotocol.Policy) (map[string]json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	if policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat {
		return nil, validateChatExecutionFields(policy, fields)
	}
	// A case variant of a modelled name would decode into the modelled
	// field through encoding/json's case-insensitive fallback while also
	// being captured here, splitting one field into two on a re-render.
	// Reject it exactly like the strict walker does.
	folded := make(map[string]string, len(chatResponseWireFields))
	for name := range chatResponseWireFields {
		folded[strings.ToLower(name)] = name
	}
	out := make(map[string]json.RawMessage)
	total := 0
	for name, raw := range fields {
		_, modelled := chatResponseWireFields[name]
		if !modelled {
			if canonical, exists := folded[strings.ToLower(name)]; exists {
				return nil, nonCanonicalUpstreamField(fmt.Errorf("field %q is not canonical; use %q", name, canonical))
			}
		}
		if modelled && !chatResponseExecutionField(name) {
			continue
		}
		if modelled && isNullJSON(raw) {
			continue
		}
		total += len(raw)
		if total > policy.Limits.SourceEnvelopeBytes {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"upstream_body_limit",
				"preserved unknown response fields exceed the source envelope budget",
				nil,
			)
		}
		out[name] = append(json.RawMessage(nil), raw...)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func chatResponseExecutionField(name string) bool {
	_, execution := chatExecutionFieldNames[name]
	return execution
}

// An empty object is the complete supported kv-transfer marker. Provider
// additions fail decode under DisallowUnknownFields.
type chatKVTransferParamsWire struct{}

type chatStopReasonWire struct {
	Text    *string
	Integer *int64
}

func (reason *chatStopReasonWire) UnmarshalJSON(raw []byte) error {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if len(text) == 0 {
			return fmt.Errorf("chat stop reason string is invalid")
		}
		reason.Text = &text
		return nil
	}
	var integer int64
	if err := json.Unmarshal(raw, &integer); err == nil {
		reason.Integer = &integer
		return nil
	}
	return fmt.Errorf("chat stop reason must be a string or integer")
}

// Chat logprobs are private model-execution evidence. The neutral response
// codec recognizes their closed wire shape so strict provider decoding does
// not reject a valid response; Looper's confidence evaluator consumes the
// evidence separately from publishable response content and accounting.
type chatLogprobsWire struct {
	Content []chatTokenLogprobWire `json:"content"`
	Refusal []chatTokenLogprobWire `json:"refusal,omitempty"`
}

type chatTokenLogprobWire struct {
	Token       string                    `json:"token"`
	Logprob     float64                   `json:"logprob"`
	Bytes       []int64                   `json:"bytes,omitempty"`
	TopLogprobs []chatTopTokenLogprobWire `json:"top_logprobs"`
}

type chatTopTokenLogprobWire struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int64 `json:"bytes,omitempty"`
}

// Mistral puts the served tier inside usage, separately from OpenAI's
// top-level service_tier. Keep this provider extension typed and closed.
type chatUsageServiceTierWire string

func (tier *chatUsageServiceTierWire) UnmarshalJSON(raw []byte) error {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	switch value {
	case "standard", "priority":
		*tier = chatUsageServiceTierWire(value)
		return nil
	default:
		return fmt.Errorf("unsupported chat usage service tier")
	}
}

type chatUsageWire struct {
	ServiceTier             *chatUsageServiceTierWire        `json:"service_tier,omitempty"`
	PromptTokens            int64                            `json:"prompt_tokens"`
	CompletionTokens        int64                            `json:"completion_tokens"`
	TotalTokens             int64                            `json:"total_tokens"`
	ComputeUnits            json.RawMessage                  `json:"compute_units,omitempty"`
	PromptTokensDetails     *chatPromptTokensDetailsWire     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *chatCompletionTokensDetailsWire `json:"completion_tokens_details,omitempty"`
	// Provider accounting extensions: xAI reports its own billing and search
	// usage, Groq reports queue and generation timings in seconds.
	CostInUSDTicks *int64   `json:"cost_in_usd_ticks,omitempty"`
	NumSourcesUsed *int64   `json:"num_sources_used,omitempty"`
	QueueTime      *float64 `json:"queue_time,omitempty"`
	PromptTime     *float64 `json:"prompt_time,omitempty"`
	CompletionTime *float64 `json:"completion_time,omitempty"`
	TotalTime      *float64 `json:"total_time,omitempty"`
	// OpenRouter reports its own pricing and server-tool accounting beside
	// canonical token usage. Keep these fields typed at the provider boundary.
	Cost          *float64                         `json:"cost,omitempty"`
	IsBYOK        *bool                            `json:"is_byok,omitempty"`
	CostDetails   *chatOpenRouterCostDetailsWire   `json:"cost_details,omitempty"`
	ServerToolUse *chatOpenRouterServerToolUseWire `json:"server_tool_use,omitempty"`
}

type chatOpenRouterCostDetailsWire struct {
	UpstreamInferenceCost            *float64 `json:"upstream_inference_cost,omitempty"`
	UpstreamInferencePromptCost      *float64 `json:"upstream_inference_prompt_cost,omitempty"`
	UpstreamInferenceCompletionsCost *float64 `json:"upstream_inference_completions_cost,omitempty"`
	ServerToolCost                   *float64 `json:"server_tool_cost,omitempty"`
}

type chatOpenRouterServerToolUseWire struct {
	WebSearchRequests *int64 `json:"web_search_requests,omitempty"`
}

type chatPromptTokensDetailsWire struct {
	CachedTokens        *int64           `json:"cached_tokens,omitempty"`
	CacheWriteTokens    *int64           `json:"cache_write_tokens,omitempty"`
	CreatedCacheTokens  *int64           `json:"created_cache_tokens,omitempty"`
	CacheCreationTokens *int64           `json:"cache_creation_tokens,omitempty"`
	MultimodalTokens    map[string]int64 `json:"multimodal_tokens,omitempty"`
	AudioTokens         int64            `json:"audio_tokens,omitempty"`
	TextTokens          int64            `json:"text_tokens,omitempty"`
	ImageTokens         int64            `json:"image_tokens,omitempty"`
	VideoTokens         *int64           `json:"video_tokens,omitempty"`
}

type chatCompletionTokensDetailsWire struct {
	AcceptedPredictionTokens int64  `json:"accepted_prediction_tokens,omitempty"`
	AudioTokens              int64  `json:"audio_tokens,omitempty"`
	ReasoningTokens          int64  `json:"reasoning_tokens"`
	TextTokens               int64  `json:"text_tokens,omitempty"`
	ImageTokens              *int64 `json:"image_tokens,omitempty"`
	RejectedPredictionTokens int64  `json:"rejected_prediction_tokens,omitempty"`
}

type chatErrorWire struct {
	Message string          `json:"message"`
	Type    string          `json:"type,omitempty"`
	Param   string          `json:"param,omitempty"`
	Code    openAIErrorCode `json:"code,omitempty"`
}

func validateChatExecutionMetadata(systemFingerprint *string) error {
	if systemFingerprint != nil && (len(*systemFingerprint) == 0 || len(*systemFingerprint) > 256) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_upstream_execution_metadata",
			"upstream Chat system fingerprint is invalid",
			nil,
		)
	}
	return nil
}

func validateChatChoiceExtensions(choice chatChoiceWire, policy llmprotocol.Policy) error {
	if err := validateChatOutputExtensions(choice.Message.Audio, choice.Message.LegacyFunctionCall, policy); err != nil {
		return err
	}
	return validateChatExecutionField("routed_experts", choice.RoutedExperts, policy)
}

func validateChatChunkChoiceExtensions(choice chatChunkChoiceWire, policy llmprotocol.Policy) error {
	// Neutral stream events intentionally have no token-logprob evidence slot.
	// Under the strict policy reject non-null evidence instead of silently
	// dropping it during the provider-to-public stream rewrite; the
	// preserve-same-format policy forwards the raw chunk instead (see
	// chatStreamDecoder.pushFrame).
	if choice.Logprobs != nil && policy.UnknownFields == llmprotocol.UnknownReject {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unsupported_upstream_stream_logprobs",
			"upstream streamed Chat log probabilities are unsupported",
			nil,
		)
	}
	if err := validateChatOutputExtensions(choice.Delta.Audio, choice.Delta.LegacyFunctionCall, policy); err != nil {
		return err
	}
	return validateChatExecutionField("routed_experts", choice.RoutedExperts, policy)
}

func validateChatOutputExtensions(
	audio *chatAudioOutputWire,
	legacyFunctionCall *chatLegacyCallWire,
	policy llmprotocol.Policy,
) error {
	// Output audio has no neutral slot. Strict decoding rejects it;
	// preserve-same-format accepts it so an unchanged response replays its
	// source bytes and a re-rendered one can still drop it with diagnostics.
	if audio != nil && policy.UnknownFields == llmprotocol.UnknownReject {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unsupported_upstream_audio",
			"upstream Chat output audio is unsupported",
			nil,
		)
	}
	if legacyFunctionCall != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unsupported_upstream_function_call",
			"upstream legacy Chat function calls are unsupported",
			nil,
		)
	}
	return nil
}
