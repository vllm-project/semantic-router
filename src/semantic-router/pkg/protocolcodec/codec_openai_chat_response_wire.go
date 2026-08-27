package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type chatResponseWire struct {
	ID                string                    `json:"id"`
	Object            string                    `json:"object,omitempty"`
	Created           int64                     `json:"created,omitempty"`
	Model             string                    `json:"model"`
	Choices           []chatChoiceWire          `json:"choices"`
	Usage             *chatUsageWire            `json:"usage,omitempty"`
	Metadata          map[string]string         `json:"metadata,omitempty"`
	Moderation        json.RawMessage           `json:"moderation,omitempty"`
	Error             *chatErrorWire            `json:"error,omitempty"`
	ServiceTier       *chatServiceTierWire      `json:"service_tier,omitempty"`
	SystemFingerprint *string                   `json:"system_fingerprint,omitempty"`
	PromptLogprobs    *chatNullOnlyWire         `json:"prompt_logprobs,omitempty"`
	PromptTokenIDs    []int64                   `json:"prompt_token_ids,omitempty"`
	PromptText        *chatNullOnlyWire         `json:"prompt_text,omitempty"`
	KVTransferParams  *chatKVTransferParamsWire `json:"kv_transfer_params,omitempty"`
	ECTransferParams  *chatNullOnlyWire         `json:"ec_transfer_params,omitempty"`
	Metrics           *chatNullOnlyWire         `json:"metrics,omitempty"`
}

type chatChoiceWire struct {
	Index         int                 `json:"index"`
	Message       chatMessageWire     `json:"message"`
	FinishReason  *string             `json:"finish_reason"`
	Logprobs      *chatLogprobsWire   `json:"logprobs,omitempty"`
	StopReason    *chatStopReasonWire `json:"stop_reason,omitempty"`
	TokenIDs      []int64             `json:"token_ids,omitempty"`
	RoutedExperts *chatNullOnlyWire   `json:"routed_experts,omitempty"`
}

type chatServiceTierWire string

func (tier *chatServiceTierWire) UnmarshalJSON(raw []byte) error {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	switch value {
	case "auto", "default", "flex", "priority", "scale":
		*tier = chatServiceTierWire(value)
		return nil
	default:
		return fmt.Errorf("unsupported chat service tier")
	}
}

// chatNullOnlyWire closes provider fields which this protocol version only
// observes as null. A future non-null representation must be modeled before it
// can cross the strict provider boundary.
type chatNullOnlyWire struct{}

func (*chatNullOnlyWire) UnmarshalJSON(raw []byte) error {
	if !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return fmt.Errorf("chat execution field must be null")
	}
	return nil
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
		if len(text) == 0 || len(text) > 128 {
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

type chatUsageWire struct {
	PromptTokens            int64                            `json:"prompt_tokens"`
	CompletionTokens        int64                            `json:"completion_tokens"`
	TotalTokens             int64                            `json:"total_tokens"`
	ComputeUnits            json.RawMessage                  `json:"compute_units,omitempty"`
	PromptTokensDetails     *chatPromptTokensDetailsWire     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *chatCompletionTokensDetailsWire `json:"completion_tokens_details,omitempty"`
}

type chatPromptTokensDetailsWire struct {
	CachedTokens int64 `json:"cached_tokens"`
	AudioTokens  int64 `json:"audio_tokens,omitempty"`
}

type chatCompletionTokensDetailsWire struct {
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens,omitempty"`
	AudioTokens              int64 `json:"audio_tokens,omitempty"`
	ReasoningTokens          int64 `json:"reasoning_tokens"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens,omitempty"`
}

type chatErrorWire struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
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

func validateChatChoiceExtensions(choice chatChoiceWire) error {
	return validateChatOutputExtensions(choice.Message.Audio, choice.Message.LegacyFunctionCall)
}

func validateChatChunkChoiceExtensions(choice chatChunkChoiceWire) error {
	// Neutral stream events intentionally have no token-logprob evidence slot.
	// Reject non-null evidence instead of silently dropping it during the
	// provider-to-public stream rewrite.
	if choice.Logprobs != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unsupported_upstream_stream_logprobs",
			"upstream streamed Chat log probabilities are unsupported",
			nil,
		)
	}
	return validateChatOutputExtensions(choice.Delta.Audio, choice.Delta.LegacyFunctionCall)
}

func validateChatOutputExtensions(
	audio *chatAudioOutputWire,
	legacyFunctionCall *chatLegacyCallWire,
) error {
	if audio != nil {
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
