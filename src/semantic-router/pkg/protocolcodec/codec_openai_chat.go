package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

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
	Name               string    `json:"name"`
	Arguments          string    `json:"arguments,omitempty"`
	TokenizedArguments *[]string `json:"TokenizedArguments,omitempty"`
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
	if err := validateChatRequestWire(wire); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
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

func validateChatRequestWire(wire chatRequestWire) error {
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"prompt_cache_key": wire.PromptCacheKey, "prompt_cache_retention": wire.PromptCacheRetention,
		"prompt_cache_options": wire.PromptCacheOptions, "safety_identifier": wire.SafetyIdentifier,
		"audio": wire.Audio, "function_call": wire.FunctionCall, "functions": wire.Functions,
		"logit_bias": wire.LogitBias, "logprobs": wire.Logprobs, "modalities": wire.Modalities,
		"moderation": wire.Moderation, "prediction": wire.Prediction, "service_tier": wire.ServiceTier,
		"top_logprobs": wire.TopLogprobs, "verbosity": wire.Verbosity,
		"web_search_options": wire.WebSearchOptions,
	}); err != nil {
		return err
	}
	if len(wire.Messages) == 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"messages_required",
			"messages must contain at least one item",
			nil,
		)
	}
	if wire.MaxTokens != nil && *wire.MaxTokens < 0 ||
		wire.MaxCompletionTokens != nil && *wire.MaxCompletionTokens < 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_chat_max_output_tokens",
			"Chat Completions output token limit cannot be negative",
			nil,
		)
	}
	if wire.MaxTokens != nil && wire.MaxCompletionTokens != nil && *wire.MaxTokens != *wire.MaxCompletionTokens {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"conflicting_max_output_tokens",
			"max_tokens and max_completion_tokens cannot specify different limits",
			nil,
		)
	}
	return nil
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
	for _, call := range wire.ToolCalls {
		if call.Function.TokenizedArguments != nil {
			return llmprotocol.Message{}, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				"unsupported_messages_tool_calls_function_tokenized_arguments",
				"messages.tool_calls.function.TokenizedArguments is provider execution metadata and is not accepted from clients",
				nil,
			)
		}
	}
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
	if choice, found := decodeChatToolChoiceMode(raw); found {
		return choice, nil
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

func decodeChatToolChoiceMode(raw json.RawMessage) (llmprotocol.ToolChoice, bool) {
	var mode string
	if json.Unmarshal(raw, &mode) != nil {
		return llmprotocol.ToolChoice{}, false
	}
	switch mode {
	case "auto":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}, true
	case "none":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}, true
	case "required":
		return llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceRequired}, true
	default:
		return llmprotocol.ToolChoice{}, false
	}
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
