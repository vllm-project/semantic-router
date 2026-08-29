package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type OpenAIChatCodec struct{}

func (OpenAIChatCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIChatV1 }
func (OpenAIChatCodec) Stateless() bool                { return true }
func (OpenAIChatCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityAudioInput,
		llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStructuredJSON, llmprotocol.CapabilityStrictJSONSchema, llmprotocol.CapabilityStrictToolSchema,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
		llmprotocol.CapabilityMultipleCandidates, llmprotocol.CapabilityCacheDirectives,
		llmprotocol.CapabilityReasoningEffort,
		llmprotocol.CapabilityReasoningBudget, llmprotocol.CapabilitySamplingSeed,
		llmprotocol.CapabilitySamplingPenalties, llmprotocol.CapabilityStopSequences,
		llmprotocol.CapabilityRequestMetadata, llmprotocol.CapabilityRequestStorage,
	)
}

type chatRequestWire struct {
	Model                string                 `json:"model"`
	Messages             []chatMessageWire      `json:"messages"`
	Tools                []chatToolWire         `json:"tools,omitempty"`
	ToolChoice           json.RawMessage        `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                  `json:"parallel_tool_calls,omitempty"`
	CandidateCount       *int64                 `json:"n,omitempty"`
	Temperature          *float64               `json:"temperature,omitempty"`
	TopP                 *float64               `json:"top_p,omitempty"`
	MaxTokens            *int64                 `json:"max_tokens,omitempty"`
	MaxCompletionTokens  *int64                 `json:"max_completion_tokens,omitempty"`
	Seed                 *int64                 `json:"seed,omitempty"`
	FrequencyPenalty     *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty      *float64               `json:"presence_penalty,omitempty"`
	Stop                 json.RawMessage        `json:"stop,omitempty"`
	ResponseFormat       *chatOutputWire        `json:"response_format,omitempty"`
	ReasoningEffort      string                 `json:"reasoning_effort,omitempty"`
	ReasoningBudget      *int64                 `json:"reasoning_budget_tokens,omitempty"`
	Stream               bool                   `json:"stream,omitempty"`
	StreamOptions        *chatStreamOptionsWire `json:"stream_options,omitempty"`
	Metadata             map[string]string      `json:"metadata,omitempty"`
	Store                *bool                  `json:"store,omitempty"`
	User                 string                 `json:"user,omitempty"`
	PromptCacheKey       json.RawMessage        `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention json.RawMessage        `json:"prompt_cache_retention,omitempty"`
	PromptCacheOptions   json.RawMessage        `json:"prompt_cache_options,omitempty"`
	SafetyIdentifier     json.RawMessage        `json:"safety_identifier,omitempty"`
	Audio                json.RawMessage        `json:"audio,omitempty"`
	FunctionCall         json.RawMessage        `json:"function_call,omitempty"`
	Functions            json.RawMessage        `json:"functions,omitempty"`
	LogitBias            json.RawMessage        `json:"logit_bias,omitempty"`
	Logprobs             json.RawMessage        `json:"logprobs,omitempty"`
	Modalities           json.RawMessage        `json:"modalities,omitempty"`
	Moderation           json.RawMessage        `json:"moderation,omitempty"`
	Prediction           json.RawMessage        `json:"prediction,omitempty"`
	ServiceTier          json.RawMessage        `json:"service_tier,omitempty"`
	TopLogprobs          json.RawMessage        `json:"top_logprobs,omitempty"`
	Verbosity            json.RawMessage        `json:"verbosity,omitempty"`
	WebSearchOptions     json.RawMessage        `json:"web_search_options,omitempty"`
}

type chatStreamOptionsWire struct {
	IncludeUsage       *bool `json:"include_usage,omitempty"`
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

type chatMessageWire struct {
	ID                 string               `json:"id,omitempty"`
	Role               string               `json:"role"`
	Content            json.RawMessage      `json:"content,omitempty"`
	Refusal            *string              `json:"refusal,omitempty"`
	Reasoning          *string              `json:"reasoning_content,omitempty"`
	AlternateReasoning *string              `json:"reasoning,omitempty"`
	Audio              *chatAudioOutputWire `json:"audio,omitempty"`
	LegacyFunctionCall *chatLegacyCallWire  `json:"function_call,omitempty"`
	ToolCalls          []chatToolCallWire   `json:"tool_calls,omitempty"`
	ToolCallID         string               `json:"tool_call_id,omitempty"`
	Annotations        []chatAnnotationWire `json:"annotations,omitempty"`
	Name               json.RawMessage      `json:"name,omitempty"`
}

type chatAudioOutputWire struct {
	ID         string `json:"id"`
	Data       string `json:"data"`
	ExpiresAt  int64  `json:"expires_at"`
	Transcript string `json:"transcript"`
}

type chatLegacyCallWire struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatAnnotationWire struct {
	Type        string                         `json:"type"`
	URLCitation *chatURLCitationAnnotationWire `json:"url_citation,omitempty"`
}

type chatURLCitationAnnotationWire struct {
	URL        string `json:"url"`
	Title      string `json:"title,omitempty"`
	StartIndex int64  `json:"start_index"`
	EndIndex   int64  `json:"end_index"`
}

type chatContentWire struct {
	Type                  string                     `json:"type"`
	Text                  string                     `json:"text,omitempty"`
	Refusal               string                     `json:"refusal,omitempty"`
	ImageURL              *chatImageURLWire          `json:"image_url,omitempty"`
	InputAudio            *chatInputAudioWire        `json:"input_audio,omitempty"`
	File                  *chatFileWire              `json:"file,omitempty"`
	CacheControl          *anthropicCacheControlWire `json:"cache_control,omitempty"`
	PromptCacheBreakpoint json.RawMessage            `json:"prompt_cache_breakpoint,omitempty"`
}

type chatFileWire struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type chatImageURLWire struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatInputAudioWire struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type chatToolCallWire struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function chatFunctionCallWire `json:"function"`
	Custom   json.RawMessage      `json:"custom,omitempty"`
}

type chatFunctionCallWire struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}

type chatFunctionDefinitionWire struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type chatToolWire struct {
	Type         string                     `json:"type"`
	Function     chatFunctionDefinitionWire `json:"function"`
	CacheControl *anthropicCacheControlWire `json:"cache_control,omitempty"`
}

type chatOutputWire struct {
	Type       string          `json:"type"`
	JSONObject json.RawMessage `json:"json_schema,omitempty"`
}

func (OpenAIChatCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire chatRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"prompt_cache_key": wire.PromptCacheKey, "prompt_cache_retention": wire.PromptCacheRetention,
		"prompt_cache_options": wire.PromptCacheOptions, "safety_identifier": wire.SafetyIdentifier,
		"audio": wire.Audio, "function_call": wire.FunctionCall, "functions": wire.Functions,
		"logit_bias": wire.LogitBias, "logprobs": wire.Logprobs, "modalities": wire.Modalities,
		"moderation": wire.Moderation, "prediction": wire.Prediction, "service_tier": wire.ServiceTier,
		"top_logprobs": wire.TopLogprobs, "verbosity": wire.Verbosity,
		"web_search_options": wire.WebSearchOptions,
	}); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if len(wire.Messages) == 0 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"messages_required",
			"messages must contain at least one item",
			nil,
		)
	}
	if wire.MaxTokens != nil && *wire.MaxTokens < 0 ||
		wire.MaxCompletionTokens != nil && *wire.MaxCompletionTokens < 0 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_chat_max_output_tokens",
			"Chat Completions output token limit cannot be negative",
			nil,
		)
	}
	if wire.MaxTokens != nil && wire.MaxCompletionTokens != nil && *wire.MaxTokens != *wire.MaxCompletionTokens {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"conflicting_max_output_tokens",
			"max_tokens and max_completion_tokens cannot specify different limits",
			nil,
		)
	}
	request := decodeChatBaseRequest(wire)
	if err := decodeChatMessages(wire.Messages, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeChatTools(wire.Tools, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeChatRequestOptions(wire, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.OpenAIChatV1, body, request.Generation, policy), nil, nil
}

func decodeChatBaseRequest(wire chatRequestWire) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		EndUserID: wire.User, Store: wire.Store,
		ParallelToolCalls: wire.ParallelToolCalls, CandidateCount: wire.CandidateCount,
		ReasoningEffort: wire.ReasoningEffort, ReasoningBudgetTokens: wire.ReasoningBudget,
		Sampling: llmprotocol.Sampling{
			Temperature: wire.Temperature, TopP: wire.TopP, Seed: wire.Seed,
			FrequencyPenalty: wire.FrequencyPenalty, PresencePenalty: wire.PresencePenalty,
		},
		Trusted: llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIChatV1},
	}
	if wire.StreamOptions != nil {
		request.StreamOptions = llmprotocol.StreamOptions{
			IncludeUsage:       wire.StreamOptions.IncludeUsage,
			IncludeObfuscation: wire.StreamOptions.IncludeObfuscation,
		}
	}
	if wire.MaxCompletionTokens != nil {
		request.Sampling.MaxOutputTokens = wire.MaxCompletionTokens
	} else {
		request.Sampling.MaxOutputTokens = wire.MaxTokens
	}
	return request
}

func decodeChatMessages(messages []chatMessageWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	for index, messageWire := range messages {
		message, err := decodeChatRequestMessage(messageWire, index, policy)
		if err != nil {
			return err
		}
		if message.Role == llmprotocol.RoleSystem || message.Role == llmprotocol.RoleDeveloper {
			request.Instructions = append(request.Instructions, llmprotocol.InstructionBlock{Role: message.Role, Content: message.Content})
		} else {
			request.Messages = append(request.Messages, message)
		}
	}
	return nil
}

func decodeChatTools(tools []chatToolWire, request *llmprotocol.Request) error {
	for _, toolWire := range tools {
		if toolWire.Type != "function" || strings.TrimSpace(toolWire.Function.Name) == "" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools are supported", nil)
		}
		schema := toolWire.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		request.Tools = append(request.Tools, llmprotocol.Tool{
			Name: toolWire.Function.Name, Description: toolWire.Function.Description,
			Strict: toolWire.Function.Strict, InputSchema: append(json.RawMessage(nil), schema...),
			Cache: decodeAnthropicCacheControl(toolWire.CacheControl),
		})
	}
	return nil
}

func decodeChatRequestOptions(wire chatRequestWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	stop, err := decodeStop(wire.Stop)
	if err != nil {
		return err
	}
	if len(stop) > 4 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"chat_stop_sequence_limit",
			"Chat Completions accepts at most four stop sequences",
			nil,
		)
	}
	request.Sampling.Stop = stop
	request.ToolChoice, err = decodeChatToolChoice(wire.ToolChoice, policy)
	if err != nil {
		return err
	}
	if wire.ResponseFormat != nil {
		request.OutputFormat, err = decodeChatOutputFormat(*wire.ResponseFormat)
		if err != nil {
			return err
		}
	}
	return nil
}

func decodeChatRequestMessage(wire chatMessageWire, index int, policy llmprotocol.Policy) (llmprotocol.Message, error) {
	role, err := validateChatMessageEnvelope(wire)
	if err != nil {
		return llmprotocol.Message{}, err
	}
	contents, err := decodeChatRequestMessageContent(wire.Content, role, policy)
	if err != nil {
		return llmprotocol.Message{}, err
	}
	return assembleChatMessage(wire, index, role, contents, policy)
}

func decodeChatResponseMessage(wire chatMessageWire, index int, policy llmprotocol.Policy) (llmprotocol.Message, error) {
	role, roleErr := canonicalRole(wire.Role)
	if roleErr != nil || role != llmprotocol.RoleAssistant {
		return llmprotocol.Message{}, invalidProviderResponse(
			"invalid_response_role",
			"Chat Completions response message role must be assistant",
		)
	}
	if _, err := validateChatMessageEnvelope(wire); err != nil {
		return llmprotocol.Message{}, err
	}
	contents, err := decodeChatResponseMessageContent(wire.Content)
	if err != nil {
		return llmprotocol.Message{}, err
	}
	return assembleChatMessage(wire, index, role, contents, policy)
}

func assembleChatMessage(
	wire chatMessageWire,
	index int,
	role llmprotocol.Role,
	contents []llmprotocol.Content,
	policy llmprotocol.Policy,
) (llmprotocol.Message, error) {
	message := llmprotocol.Message{ID: wire.ID, Role: role, Content: contents}
	if wire.Refusal != nil {
		message.Content = append(message.Content, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: *wire.Refusal})
	}
	reasoning := wire.Reasoning
	if reasoning == nil {
		reasoning = wire.AlternateReasoning
	}
	if reasoning != nil {
		message.Content = append(message.Content, llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: *reasoning, Reasoning: llmprotocol.ReasoningScopeText,
		})
	}
	toolCalls, err := decodeChatToolCalls(wire.ToolCalls, index, policy)
	if err != nil {
		return llmprotocol.Message{}, err
	}
	message.Content = append(message.Content, toolCalls...)
	if len(wire.Annotations) > 0 {
		if err := attachChatAnnotations(&message, wire.Annotations); err != nil {
			return llmprotocol.Message{}, err
		}
	}
	if role == llmprotocol.RoleTool {
		result := llmprotocol.ToolResult{CallID: wire.ToolCallID, Content: append([]llmprotocol.Content(nil), message.Content...)}
		message.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &result}}
	}
	return message, nil
}

func validateChatMessageEnvelope(wire chatMessageWire) (llmprotocol.Role, error) {
	if err := rejectUnsupportedRequestField("messages.name", wire.Name); err != nil {
		return "", err
	}
	if wire.Audio != nil {
		return "", llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_output_audio", "Chat output audio is unsupported", nil,
		)
	}
	if wire.LegacyFunctionCall != nil {
		return "", llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_legacy_function_call", "legacy Chat function calls are unsupported", nil,
		)
	}
	return canonicalRole(wire.Role)
}

func decodeChatRequestMessageContent(
	raw json.RawMessage,
	role llmprotocol.Role,
	policy llmprotocol.Policy,
) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
	}
	var partBodies []json.RawMessage
	if err := decodeWireValue(raw, &partBodies, policy); err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(partBodies))
	for _, partBody := range partBodies {
		var part chatContentWire
		if err := decodeWireValue(partBody, &part, policy); err != nil {
			return nil, err
		}
		if !chatRequestContentAllowed(role, part.Type) {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				"unsupported_content",
				fmt.Sprintf("Chat Completions %s message does not support %s content", role, part.Type),
				nil,
			)
		}
		content, err := decodeChatContent(partBody, part)
		if err != nil {
			return nil, err
		}
		contents = append(contents, content)
	}
	return contents, nil
}

func decodeChatResponseMessageContent(raw json.RawMessage) ([]llmprotocol.Content, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}, nil
	}
	return nil, llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"invalid_response_content",
		"Chat Completions response content must be a string or null",
		nil,
	)
}

func chatRequestContentAllowed(role llmprotocol.Role, contentType string) bool {
	switch role {
	case llmprotocol.RoleUser:
		return contentType == "text" || contentType == "image_url" || contentType == "input_audio" || contentType == "file"
	case llmprotocol.RoleAssistant:
		return contentType == "text" || contentType == "refusal"
	case llmprotocol.RoleSystem, llmprotocol.RoleDeveloper, llmprotocol.RoleTool:
		return contentType == "text"
	default:
		return false
	}
}

func decodeChatToolCalls(calls []chatToolCallWire, messageIndex int, policy llmprotocol.Policy) ([]llmprotocol.Content, error) {
	contents := make([]llmprotocol.Content, 0, len(calls))
	for toolIndex, call := range calls {
		if (call.Type != "" && call.Type != "function") || len(call.Custom) > 0 {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				"unsupported_tool_call",
				"only function tool calls enter the model protocol",
				nil,
			)
		}
		id := call.ID
		if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			id = llmprotocol.StableID("chat", fmt.Sprint(messageIndex), fmt.Sprint(toolIndex), call.Function.Name, call.Function.Arguments)
		}
		if id == "" {
			return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "tool_call_id_required", "tool call ID is required", nil)
		}
		contents = append(contents, llmprotocol.Content{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: id, Name: call.Function.Name, Arguments: call.Function.Arguments},
		})
	}
	return contents, nil
}

func attachChatAnnotations(message *llmprotocol.Message, annotations []chatAnnotationWire) error {
	citations, err := decodeChatAnnotations(annotations)
	if err != nil {
		return err
	}
	for contentIndex := range message.Content {
		if message.Content[contentIndex].Kind == llmprotocol.ContentText {
			message.Content[contentIndex].Citations = citations
			return nil
		}
	}
	return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "citation_text_required", "URL citations require text content", nil)
}

func decodeChatAnnotations(wire []chatAnnotationWire) ([]llmprotocol.Citation, error) {
	citations := make([]llmprotocol.Citation, 0, len(wire))
	for _, annotation := range wire {
		if annotation.Type != "url_citation" || annotation.URLCitation == nil {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_annotation", "Chat Completions annotation is unsupported", nil)
		}
		citations = append(citations, llmprotocol.Citation{
			URL: annotation.URLCitation.URL, Title: annotation.URLCitation.Title,
			StartIndex: annotation.URLCitation.StartIndex, EndIndex: annotation.URLCitation.EndIndex,
		})
	}
	return citations, nil
}

func encodeChatAnnotations(citations []llmprotocol.Citation) []chatAnnotationWire {
	annotations := make([]chatAnnotationWire, 0, len(citations))
	for _, citation := range citations {
		annotations = append(annotations, chatAnnotationWire{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
			URL: citation.URL, Title: citation.Title, StartIndex: citation.StartIndex, EndIndex: citation.EndIndex,
		}})
	}
	return annotations
}

func decodeChatContent(body json.RawMessage, part chatContentWire) (llmprotocol.Content, error) {
	if err := validateChatContentVariant(body, part.Type); err != nil {
		return llmprotocol.Content{}, err
	}
	if err := rejectUnsupportedRequestField("messages.content.prompt_cache_breakpoint", part.PromptCacheBreakpoint); err != nil {
		return llmprotocol.Content{}, err
	}
	switch part.Type {
	case "text":
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text, Cache: decodeAnthropicCacheControl(part.CacheControl)}, nil
	case "refusal":
		if part.CacheControl != nil {
			return llmprotocol.Content{}, unsupportedChatCacheDirective(part.Type)
		}
		text := part.Refusal
		if text == "" {
			text = part.Text
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: text}, nil
	case "image_url":
		content, err := decodeChatImageContent(part.ImageURL)
		content.Cache = decodeAnthropicCacheControl(part.CacheControl)
		return content, err
	case "input_audio":
		if part.InputAudio == nil {
			return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "audio_required", "input audio is required", nil)
		}
		return llmprotocol.Content{Kind: llmprotocol.ContentAudio, Data: part.InputAudio.Data, MediaType: part.InputAudio.Format, Cache: decodeAnthropicCacheControl(part.CacheControl)}, nil
	case "file":
		content, err := decodeChatFileContent(part.File)
		content.Cache = decodeAnthropicCacheControl(part.CacheControl)
		return content, err
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "chat content type is unsupported", nil)
	}
}

func validateChatContentVariant(body json.RawMessage, typeName string) error {
	known := []string{
		"cache_control", "file", "image_url", "input_audio", "prompt_cache_breakpoint", "refusal", "text", "type",
	}
	allowedByType := map[string][]string{
		"text":        {"cache_control", "prompt_cache_breakpoint", "text", "type"},
		"refusal":     {"refusal", "type"},
		"image_url":   {"cache_control", "image_url", "prompt_cache_breakpoint", "type"},
		"input_audio": {"cache_control", "input_audio", "prompt_cache_breakpoint", "type"},
		"file":        {"cache_control", "file", "prompt_cache_breakpoint", "type"},
	}
	allowed, recognized := allowedByType[typeName]
	if !recognized {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return err
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, name := range allowed {
		allowedSet[name] = struct{}{}
	}
	for _, name := range known {
		if _, present := object[name]; !present {
			continue
		}
		if _, valid := allowedSet[name]; valid {
			continue
		}
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_content_variant",
			"Chat Completions content includes a field from a different union variant: "+name,
			nil,
		)
	}
	return nil
}

func unsupportedChatCacheDirective(kind string) error {
	return llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"unsupported_cache_directive",
		fmt.Sprintf("Chat Completions cannot attach cache_control to %s content", kind),
		nil,
	)
}

func decodeChatImageContent(image *chatImageURLWire) (llmprotocol.Content, error) {
	if image == nil || image.URL == "" {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_url_required", "image URL is required", nil)
	}
	if mediaType, data, inline := decodeDataURL(image.URL); inline {
		return llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: mediaType, Data: data, Detail: image.Detail}, nil
	}
	return llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: image.URL, Detail: image.Detail}, nil
}

func decodeChatFileContent(file *chatFileWire) (llmprotocol.Content, error) {
	if file == nil {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "file_required", "file content is required", nil)
	}
	content := llmprotocol.Content{Kind: llmprotocol.ContentFile, FileID: file.FileID, Filename: file.Filename}
	if file.FileData != "" {
		content.MediaType, content.Data = decodeChatFileData(file.FileData)
	}
	if content.FileID == "" && content.Data == "" {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "file_source_required", "file content requires file_id or file_data", nil)
	}
	if content.FileID != "" && content.Data != "" {
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "file_source_conflict", "file content accepts exactly one source", nil)
	}
	return content, nil
}

func decodeChatFileData(raw string) (string, string) {
	if mediaType, data, inline := decodeDataURL(raw); inline {
		return mediaType, data
	}
	return "application/octet-stream", raw
}

func decodeStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_stop", "stop must be a string or string array", nil)
	}
	return many, nil
}

func decodeChatToolChoice(raw json.RawMessage, policy llmprotocol.Policy) (llmprotocol.ToolChoice, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return llmprotocol.ToolChoice{}, nil
	}
	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		switch mode {
		case "auto":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}, nil
		case "none":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}, nil
		case "required":
			return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceRequired}, nil
		}
	}
	var discriminator struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &discriminator) == nil && (discriminator.Type == "allowed_tools" || discriminator.Type == "custom") {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_tool_choice",
			"Chat Completions tool choice cannot be represented by the neutral protocol",
			nil,
		)
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if decodeWireValue(raw, &named, policy) != nil || named.Type != "function" || named.Function.Name == "" {
		return llmprotocol.ToolChoice{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
	}
	return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: named.Function.Name}, nil
}

func decodeChatOutputFormat(wire chatOutputWire) (llmprotocol.OutputFormat, error) {
	switch wire.Type {
	case "", "text":
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputText}, nil
	case "json_object":
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONObject}, nil
	case "json_schema":
		var schema struct {
			Name        string          `json:"name"`
			Description string          `json:"description,omitempty"`
			Strict      *bool           `json:"strict,omitempty"`
			Schema      json.RawMessage `json:"schema"`
		}
		if json.Unmarshal(wire.JSONObject, &schema) != nil || len(schema.Schema) == 0 {
			return llmprotocol.OutputFormat{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json_schema", "response JSON Schema is invalid", nil)
		}
		return llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONSchema, Name: schema.Name, Description: schema.Description, Strict: schema.Strict, Schema: schema.Schema}, nil
	default:
		return llmprotocol.OutputFormat{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "response format is unsupported", nil)
	}
}

func (OpenAIChatCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	if request.ReasoningDisplay != "" {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_reasoning_display",
			"Chat Completions cannot represent reasoning display controls",
			nil,
		)
	}
	if request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens < 0 {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_chat_max_output_tokens",
			"Chat Completions cannot represent a negative output token limit",
			nil,
		)
	}
	if len(request.Sampling.Stop) > 4 {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_chat_stop_sequence_limit",
			"Chat Completions cannot represent more than four stop sequences",
			nil,
		)
	}
	diagnostics, validationErr := chatRequestDiagnostics(request, policy)
	if validationErr != nil {
		return nil, diagnostics, validationErr
	}
	wire := encodeChatBaseRequest(request)
	if encodeErr := appendChatMessages(&wire, request); encodeErr != nil {
		return nil, diagnostics, encodeErr
	}
	if len(wire.Messages) == 0 {
		return nil, diagnostics, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"chat_messages_required",
			"Chat Completions requires at least one conversation message",
			nil,
		)
	}
	appendChatTools(&wire, request.Tools)
	if encodeErr := encodeChatRequestOptions(&wire, request); encodeErr != nil {
		return nil, diagnostics, encodeErr
	}
	body, encodeErr := marshalWire(wire)
	return body, diagnostics, encodeErr
}

func chatRequestDiagnostics(request llmprotocol.Request, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if request.PreviousResponseID == "" && request.ConversationID == "" && request.Truncation == "" {
		return diagnostics, nil
	}
	err := appendLossy(
		&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIChatV1,
		"conversation_state", "Chat Completions has no stateful response reference",
	)
	return diagnostics, err
}

func encodeChatBaseRequest(request llmprotocol.Request) chatRequestWire {
	wire := chatRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		Store: request.Store, User: request.EndUserID,
		ParallelToolCalls: request.ParallelToolCalls, CandidateCount: request.CandidateCount,
		Temperature: request.Sampling.Temperature, TopP: request.Sampling.TopP,
		MaxCompletionTokens: request.Sampling.MaxOutputTokens, Seed: request.Sampling.Seed,
		FrequencyPenalty: request.Sampling.FrequencyPenalty, PresencePenalty: request.Sampling.PresencePenalty,
		ReasoningEffort: request.ReasoningEffort, ReasoningBudget: request.ReasoningBudgetTokens,
	}
	if request.Stream && (request.StreamOptions.IncludeUsage != nil || request.StreamOptions.IncludeObfuscation != nil) {
		wire.StreamOptions = &chatStreamOptionsWire{
			IncludeUsage:       request.StreamOptions.IncludeUsage,
			IncludeObfuscation: request.StreamOptions.IncludeObfuscation,
		}
	}
	return wire
}

func appendChatMessages(wire *chatRequestWire, request llmprotocol.Request) error {
	for _, instruction := range request.Instructions {
		encoded, err := encodeChatMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content})
		if err != nil {
			return err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	for _, message := range request.Messages {
		encoded, err := encodeChatMessage(message)
		if err != nil {
			return err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	return nil
}

func appendChatTools(wire *chatRequestWire, tools []llmprotocol.Tool) {
	for _, tool := range tools {
		wire.Tools = append(wire.Tools, chatToolWire{Type: "function", Function: chatFunctionDefinitionWire{
			Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict,
		}, CacheControl: encodeAnthropicCacheControl(tool.Cache)})
	}
}

func encodeChatRequestOptions(wire *chatRequestWire, request llmprotocol.Request) error {
	wire.ToolChoice = encodeChatToolChoice(request.ToolChoice)
	if len(request.Sampling.Stop) == 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop[0])
	} else if len(request.Sampling.Stop) > 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop)
	}
	output, err := encodeChatOutputFormat(request.OutputFormat)
	if err != nil {
		return err
	}
	wire.ResponseFormat = output
	return nil
}

func encodeChatMessage(message llmprotocol.Message) (chatMessageWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return chatMessageWire{}, err
	}
	wire := chatMessageWire{ID: message.ID, Role: role}
	state := chatMessageEncodingState{wire: &wire, parts: make([]chatContentWire, 0, len(message.Content))}
	for _, content := range message.Content {
		if err := state.appendContent(content); err != nil {
			return chatMessageWire{}, err
		}
	}
	if state.citationTextBlocks > 1 {
		return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "citation_text_ambiguous", "Chat Completions citations require one text block", nil)
	}
	if len(state.parts) == 1 && state.parts[0].Type == "text" && state.parts[0].CacheControl == nil {
		wire.Content, _ = json.Marshal(state.parts[0].Text)
	} else if len(state.parts) > 0 {
		wire.Content, _ = json.Marshal(state.parts)
	}
	return wire, nil
}

type chatMessageEncodingState struct {
	wire               *chatMessageWire
	parts              []chatContentWire
	citationTextBlocks int
}

func (state *chatMessageEncodingState) appendContent(content llmprotocol.Content) error {
	switch content.Kind {
	case llmprotocol.ContentText:
		state.appendText(content)
	case llmprotocol.ContentRefusal:
		if content.Cache != nil {
			return unsupportedChatCacheDirective(string(content.Kind))
		}
		if state.wire.Refusal == nil {
			state.wire.Refusal = new(string)
		}
		*state.wire.Refusal += content.Text
	case llmprotocol.ContentReasoning:
		if content.Cache != nil {
			return unsupportedChatCacheDirective(string(content.Kind))
		}
		if state.wire.Reasoning == nil {
			state.wire.Reasoning = new(string)
		}
		*state.wire.Reasoning += content.Text
	case llmprotocol.ContentImage:
		return state.appendImage(content)
	case llmprotocol.ContentAudio:
		state.parts = append(state.parts, chatContentWire{Type: "input_audio", InputAudio: &chatInputAudioWire{Data: content.Data, Format: content.MediaType}, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	case llmprotocol.ContentFile:
		return state.appendFile(content)
	case llmprotocol.ContentToolCall:
		if content.Cache != nil {
			return unsupportedChatCacheDirective(string(content.Kind))
		}
		return state.appendToolCall(content.ToolCall)
	case llmprotocol.ContentToolResult:
		if content.Cache != nil {
			return unsupportedChatCacheDirective(string(content.Kind))
		}
		return state.appendToolResult(content.ToolResult)
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as chat", nil)
	}
	return nil
}

func (state *chatMessageEncodingState) appendText(content llmprotocol.Content) {
	state.parts = append(state.parts, chatContentWire{Type: "text", Text: content.Text, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	if len(content.Citations) > 0 {
		state.citationTextBlocks++
		state.wire.Annotations = append(state.wire.Annotations, encodeChatAnnotations(content.Citations)...)
	}
}

func (state *chatMessageEncodingState) appendImage(content llmprotocol.Content) error {
	if content.FileID != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "image_file_id", "Chat Completions cannot encode image file IDs", nil)
	}
	imageURL := content.URL
	if content.Data != "" {
		if !strings.HasPrefix(content.MediaType, "image/") {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_media_type", "inline Chat images require an image media type", nil)
		}
		imageURL = "data:" + content.MediaType + ";base64," + content.Data
	}
	if imageURL == "" {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_source", "Chat images require a URL or inline data", nil)
	}
	state.parts = append(state.parts, chatContentWire{Type: "image_url", ImageURL: &chatImageURLWire{URL: imageURL, Detail: content.Detail}, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	return nil
}

func (state *chatMessageEncodingState) appendFile(content llmprotocol.Content) error {
	if content.Detail != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_detail", "Chat Completions cannot encode file detail", nil)
	}
	if content.URL != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_url", "Chat Completions cannot encode file URLs", nil)
	}
	file := &chatFileWire{Filename: content.Filename, FileID: content.FileID}
	if content.Data != "" {
		if content.MediaType == "" {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "file_media_type", "inline Chat files require a media type", nil)
		}
		file.FileData = "data:" + content.MediaType + ";base64," + content.Data
	}
	state.parts = append(state.parts, chatContentWire{Type: "file", File: file, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	return nil
}

func (state *chatMessageEncodingState) appendToolCall(call *llmprotocol.ToolCall) error {
	if call == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call content is invalid", nil)
	}
	state.wire.ToolCalls = append(state.wire.ToolCalls, chatToolCallWire{
		ID: call.ID, Type: "function",
		Function: chatFunctionCallWire{Name: call.Name, Arguments: call.Arguments},
	})
	return nil
}

func (state *chatMessageEncodingState) appendToolResult(result *llmprotocol.ToolResult) error {
	if result == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result content is invalid", nil)
	}
	state.wire.ToolCallID = result.CallID
	for _, resultContent := range result.Content {
		if resultContent.Kind != llmprotocol.ContentText {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "tool_result_media", "chat tool results support text only", nil)
		}
		state.parts = append(state.parts, chatContentWire{
			Type: "text", Text: resultContent.Text,
			CacheControl: encodeAnthropicCacheControl(resultContent.Cache),
		})
	}
	return nil
}

func encodeChatToolChoice(choice llmprotocol.ToolChoice) json.RawMessage {
	switch choice.Mode {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
		body, _ := json.Marshal(choice.Mode)
		return body
	case llmprotocol.ToolChoiceNamed:
		body, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]string{"name": choice.Name}})
		return body
	default:
		return nil
	}
}

func encodeChatOutputFormat(format llmprotocol.OutputFormat) (*chatOutputWire, error) {
	switch format.Kind {
	case "", llmprotocol.OutputText:
		return nil, nil
	case llmprotocol.OutputJSONObject:
		return &chatOutputWire{Type: "json_object"}, nil
	case llmprotocol.OutputJSONSchema:
		body, err := json.Marshal(map[string]any{"name": format.Name, "description": format.Description, "strict": format.Strict, "schema": format.Schema})
		return &chatOutputWire{Type: "json_schema", JSONObject: body}, err
	default:
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "output format cannot be encoded as chat", nil)
	}
}

func (OpenAIChatCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire chatResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateChatResponseResource(wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateChatExecutionMetadata(wire.SystemFingerprint); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response := decodeChatResponseEnvelope(wire)
	var diagnostics llmprotocol.Diagnostics
	appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIChatV1, map[string]bool{
		"metadata": len(wire.Metadata) > 0, "moderation": len(wire.Moderation) > 0,
	}, "response request metadata is not model output")
	if err := decodeChatChoices(wire, &response, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
	}
	if wire.Usage != nil {
		response.Usage = decodeChatUsage(*wire.Usage)
		appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIChatV1, map[string]bool{
			"usage.compute_units":                                        len(wire.Usage.ComputeUnits) > 0,
			"usage.prompt_tokens_details.audio_tokens":                   wire.Usage.PromptTokensDetails != nil && wire.Usage.PromptTokensDetails.AudioTokens != 0,
			"usage.prompt_tokens_details.image_tokens":                   wire.Usage.PromptTokensDetails != nil && wire.Usage.PromptTokensDetails.ImageTokens != 0,
			"usage.prompt_tokens_details.text_tokens":                    wire.Usage.PromptTokensDetails != nil && wire.Usage.PromptTokensDetails.TextTokens != 0,
			"usage.completion_tokens_details.accepted_prediction_tokens": wire.Usage.CompletionTokensDetails != nil && wire.Usage.CompletionTokensDetails.AcceptedPredictionTokens != 0,
			"usage.completion_tokens_details.audio_tokens":               wire.Usage.CompletionTokensDetails != nil && wire.Usage.CompletionTokensDetails.AudioTokens != 0,
			"usage.completion_tokens_details.rejected_prediction_tokens": wire.Usage.CompletionTokensDetails != nil && wire.Usage.CompletionTokensDetails.RejectedPredictionTokens != 0,
			"usage.completion_tokens_details.text_tokens":                wire.Usage.CompletionTokensDetails != nil && wire.Usage.CompletionTokensDetails.TextTokens != 0,
		}, "provider accounting detail has no separate protocol-neutral bucket")
	}
	envelope := responseEnvelope(llmprotocol.OpenAIChatV1, body, response.Generation, response.SourceStopReason, policy)
	return response, envelope, diagnostics, nil
}

func decodeChatResponseEnvelope(wire chatResponseWire) llmprotocol.Response {
	response := llmprotocol.Response{
		Generation: 1, ID: wire.ID, Model: wire.Model,
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
	if wire.Created > 0 {
		response.CreatedAt = time.Unix(wire.Created, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{
			Category: decodeProviderErrorCategory(wire.Error.Type, wire.Error.Code),
			Code:     wire.Error.Code, Message: wire.Error.Message, Parameter: wire.Error.Param,
		}
		response.StopReason = llmprotocol.StopError
	}
	return response
}

func decodeChatChoices(wire chatResponseWire, response *llmprotocol.Response, policy llmprotocol.Policy) error {
	choices, err := normalizedChatChoices(wire.Choices)
	if err != nil {
		return err
	}
	for choiceIndex, choice := range choices {
		item, err := decodeChatChoiceItem(choice, wire.ID, policy)
		if err != nil {
			return err
		}
		if choiceIndex == 0 {
			response.Output = append(response.Output, item)
			response.Evidence.TokenLogprobs = decodeChatTokenLogprobs(choice.Logprobs)
			if choice.FinishReason != nil {
				response.SourceStopReason = *choice.FinishReason
				response.StopReason = decodeChatStop(*choice.FinishReason)
			}
			continue
		}
		response.Alternatives = append(response.Alternatives, []llmprotocol.OutputItem{item})
	}
	return nil
}

func decodeChatChoiceItem(choice chatChoiceWire, responseID string, policy llmprotocol.Policy) (llmprotocol.OutputItem, error) {
	if err := validateChatChoiceExtensions(choice); err != nil {
		return llmprotocol.OutputItem{}, err
	}
	message, err := decodeChatResponseMessage(choice.Message, choice.Index, policy)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	item := llmprotocol.OutputItem(message)
	if item.ID == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		item.ID = llmprotocol.StableID("chat-response", responseID, fmt.Sprint(choice.Index))
	}
	return item, nil
}

func decodeChatTokenLogprobs(wire *chatLogprobsWire) []llmprotocol.TokenLogprob {
	if wire == nil {
		return nil
	}
	tokens := make([]llmprotocol.TokenLogprob, 0, len(wire.Content))
	for _, token := range wire.Content {
		decoded := llmprotocol.TokenLogprob{Token: token.Token, Logprob: token.Logprob}
		for _, alternative := range token.TopLogprobs {
			decoded.Alternatives = append(decoded.Alternatives, llmprotocol.TokenLogprobAlternative{
				Token: alternative.Token, Logprob: alternative.Logprob,
			})
		}
		tokens = append(tokens, decoded)
	}
	return tokens
}

func decodeChatUsage(wire chatUsageWire) llmprotocol.Usage {
	usage := llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputUncached:   unknownCount(),
		InputCacheRead:  unknownCount(),
		InputCacheWrite: unknownCount(),
		OutputReasoning: unknownCount(),
		OutputOther:     unknownCount(),
		InputTotal:      authoritative(wire.PromptTokens),
		OutputTotal:     authoritative(wire.CompletionTokens),
		Total:           authoritative(wire.TotalTokens),
	}
	if wire.PromptTokensDetails != nil {
		cached, cacheWrite := wire.PromptTokensDetails.CachedTokens, wire.PromptTokensDetails.CacheWriteTokens
		uncached := int64(-1)
		if cached >= 0 && cacheWrite >= 0 && wire.PromptTokens >= cached && cacheWrite <= wire.PromptTokens-cached {
			uncached = wire.PromptTokens - cached - cacheWrite
		}
		usage.InputCacheRead = authoritative(cached)
		usage.InputCacheWrite = authoritative(cacheWrite)
		usage.InputUncached = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(uncached), Provenance: llmprotocol.UsageDerived,
		}
	}
	if wire.CompletionTokensDetails != nil {
		reasoning := wire.CompletionTokensDetails.ReasoningTokens
		other := wire.CompletionTokens - reasoning
		if reasoning < 0 || wire.CompletionTokens < reasoning {
			other = -1
		}
		usage.OutputReasoning = authoritative(reasoning)
		usage.OutputOther = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(other), Provenance: llmprotocol.UsageDerived,
		}
	}
	return usage
}

func (OpenAIChatCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if response.Error != nil {
		var diagnostics llmprotocol.Diagnostics
		if response.Usage.State == llmprotocol.UsageAvailable {
			appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIChatV1, "usage", "Chat Completions error envelopes cannot carry token usage")
		}
		return OpenAIChatCodec{}.EncodeTransportError(llmprotocol.TransportError{Error: response.Error}), diagnostics, nil
	}
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if response.StopReason == llmprotocol.StopPaused || response.StopReason == llmprotocol.StopContextWindow || response.StopReason == llmprotocol.StopCanceled || response.StopReason == llmprotocol.StopUnknown {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIChatV1, "response.stop_reason", "Chat Completions cannot represent the source terminal reason"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := chatResponseWire{ID: response.ID, Object: "chat.completion", Model: response.Model}
	if !response.CreatedAt.IsZero() {
		wire.Created = response.CreatedAt.Unix()
	}
	primary, encodeResponseErr := encodeChatOutput(response.Output, 0, response.StopReason)
	if encodeResponseErr != nil {
		return nil, diagnostics, encodeResponseErr
	}
	wire.Choices = append(wire.Choices, primary)
	for index, alternative := range response.Alternatives {
		choice, err := encodeChatOutput(alternative, index+1, response.StopReason)
		if err != nil {
			return nil, diagnostics, err
		}
		wire.Choices = append(wire.Choices, choice)
	}
	wire.Usage = encodeChatUsage(response.Usage)
	body, encodeResponseErr := marshalWire(wire)
	return body, diagnostics, encodeResponseErr
}

func encodeChatOutput(items []llmprotocol.OutputItem, index int, stop llmprotocol.StopReason) (chatChoiceWire, error) {
	combined := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, item := range items {
		if item.Role != "" {
			combined.Role = item.Role
		}
		combined.Content = append(combined.Content, item.Content...)
	}
	message, err := encodeChatMessage(combined)
	if err != nil {
		return chatChoiceWire{}, err
	}
	reason := encodeChatStop(stop)
	return chatChoiceWire{Index: index, Message: message, FinishReason: &reason}, nil
}

func encodeChatUsage(usage llmprotocol.Usage) *chatUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	prompt := tokenValue(usage.InputTotal)
	completion := tokenValue(usage.OutputTotal)
	total := tokenValue(usage.Total)
	if total == 0 {
		total = prompt + completion
	}
	wire := &chatUsageWire{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: total}
	if usage.InputCacheRead.Value != nil || usage.InputCacheWrite.Value != nil {
		wire.PromptTokensDetails = &chatPromptTokensDetailsWire{
			CachedTokens: tokenValue(usage.InputCacheRead), CacheWriteTokens: tokenValue(usage.InputCacheWrite),
		}
	}
	if usage.OutputReasoning.Value != nil {
		wire.CompletionTokensDetails = &chatCompletionTokensDetailsWire{ReasoningTokens: tokenValue(usage.OutputReasoning)}
	}
	return wire
}

func decodeChatStop(reason string) llmprotocol.StopReason {
	switch reason {
	case "stop":
		return llmprotocol.StopEndTurn
	case "length":
		return llmprotocol.StopMaxTokens
	case "tool_calls", "function_call":
		return llmprotocol.StopToolCall
	case "content_filter":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeChatStop(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "length"
	case llmprotocol.StopToolCall:
		return "tool_calls"
	case llmprotocol.StopContentFilter:
		return "content_filter"
	default:
		return "stop"
	}
}

func (OpenAIChatCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	return decodeOpenAITransportError(body, policy)
}

func (OpenAIChatCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	return encodeOpenAITransportError(transportError)
}
