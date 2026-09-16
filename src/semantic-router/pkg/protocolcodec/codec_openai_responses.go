package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type OpenAIResponsesCodec struct{}

func (OpenAIResponsesCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIResponsesV1 }
func (OpenAIResponsesCodec) Stateless() bool                { return true }
func (OpenAIResponsesCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput, llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityImageGeneration,
		llmprotocol.CapabilityTools, llmprotocol.CapabilityParallelTools, llmprotocol.CapabilityReasoning,
		llmprotocol.CapabilityStructuredJSON, llmprotocol.CapabilityStrictJSONSchema, llmprotocol.CapabilityStrictToolSchema,
		llmprotocol.CapabilityStreaming, llmprotocol.CapabilityCacheAccounting,
		llmprotocol.CapabilityReasoningAccounting, llmprotocol.CapabilityAuthoritativeUsage,
		llmprotocol.CapabilityReasoningEffort, llmprotocol.CapabilityRequestMetadata,
		llmprotocol.CapabilityRequestStorage, llmprotocol.CapabilityAutomaticStorage,
		llmprotocol.CapabilityConversationState,
	)
}

type responsesRequestWire struct {
	Model                string                      `json:"model"`
	Input                json.RawMessage             `json:"input"`
	Instructions         json.RawMessage             `json:"instructions,omitempty"`
	Tools                json.RawMessage             `json:"tools,omitempty"`
	ToolChoice           json.RawMessage             `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                       `json:"parallel_tool_calls,omitempty"`
	Temperature          *float64                    `json:"temperature,omitempty"`
	TopP                 *float64                    `json:"top_p,omitempty"`
	MaxOutputTokens      *int64                      `json:"max_output_tokens,omitempty"`
	Metadata             map[string]string           `json:"metadata,omitempty"`
	Text                 *responsesTextWire          `json:"text,omitempty"`
	Stream               bool                        `json:"stream,omitempty"`
	Store                *bool                       `json:"store,omitempty"`
	PreviousResponseID   string                      `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage             `json:"conversation,omitempty"`
	AutoStore            *bool                       `json:"auto_store,omitempty"`
	Reasoning            *responsesReasoningWire     `json:"reasoning,omitempty"`
	Truncation           string                      `json:"truncation,omitempty"`
	User                 string                      `json:"user,omitempty"`
	Background           json.RawMessage             `json:"background,omitempty"`
	ContextManagement    json.RawMessage             `json:"context_management,omitempty"`
	Include              json.RawMessage             `json:"include,omitempty"`
	MaxToolCalls         json.RawMessage             `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage             `json:"moderation,omitempty"`
	Prompt               json.RawMessage             `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage             `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention json.RawMessage             `json:"prompt_cache_retention,omitempty"`
	PromptCacheOptions   json.RawMessage             `json:"prompt_cache_options,omitempty"`
	SafetyIdentifier     json.RawMessage             `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage             `json:"service_tier,omitempty"`
	StreamOptions        *responsesStreamOptionsWire `json:"stream_options,omitempty"`
	TopLogprobs          json.RawMessage             `json:"top_logprobs,omitempty"`
}

type responsesReasoningWire struct {
	Mode            json.RawMessage `json:"mode,omitempty"`
	Effort          string          `json:"effort,omitempty"`
	Summary         json.RawMessage `json:"summary,omitempty"`
	Context         json.RawMessage `json:"context,omitempty"`
	GenerateSummary json.RawMessage `json:"generate_summary,omitempty"`
}

type responsesStreamOptionsWire struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

type responsesTextWire struct {
	Format    responsesFormatWire `json:"format,omitempty"`
	Verbosity json.RawMessage     `json:"verbosity,omitempty"`
}

type responsesFormatWire struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type responsesToolWire struct {
	Type              string                     `json:"type"`
	Name              string                     `json:"name,omitempty"`
	Description       string                     `json:"description,omitempty"`
	Parameters        json.RawMessage            `json:"parameters,omitempty"`
	Strict            *bool                      `json:"strict,omitempty"`
	AllowedCallers    json.RawMessage            `json:"allowed_callers,omitempty"`
	DeferLoading      json.RawMessage            `json:"defer_loading,omitempty"`
	OutputSchema      json.RawMessage            `json:"output_schema,omitempty"`
	Model             string                     `json:"model,omitempty"`
	Quality           string                     `json:"quality,omitempty"`
	Size              string                     `json:"size,omitempty"`
	OutputFormat      string                     `json:"output_format,omitempty"`
	OutputCompression *int64                     `json:"output_compression,omitempty"`
	Moderation        string                     `json:"moderation,omitempty"`
	Background        string                     `json:"background,omitempty"`
	InputFidelity     string                     `json:"input_fidelity,omitempty"`
	InputImageMask    *responsesImageGenMaskWire `json:"input_image_mask,omitempty"`
	PartialImages     *int64                     `json:"partial_images,omitempty"`
	Action            string                     `json:"action,omitempty"`
}

type responsesImageGenMaskWire struct {
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type responsesItemWire struct {
	Type             string          `json:"type"`
	ID               string          `json:"id,omitempty"`
	Role             string          `json:"role,omitempty"`
	Status           string          `json:"status,omitempty"`
	Content          json.RawMessage `json:"content,omitempty"`
	Name             string          `json:"name,omitempty"`
	CallID           string          `json:"call_id,omitempty"`
	Caller           json.RawMessage `json:"caller,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
	Arguments        string          `json:"arguments,omitempty"`
	Output           json.RawMessage `json:"output,omitempty"`
	Summary          json.RawMessage `json:"summary,omitempty"`
	Phase            json.RawMessage `json:"phase,omitempty"`
	EncryptedContent json.RawMessage `json:"encrypted_content,omitempty"`
	Result           *string         `json:"result,omitempty"`
}

func (wire responsesItemWire) MarshalJSON() ([]byte, error) {
	type itemAlias responsesItemWire
	body, err := json.Marshal(itemAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "message":
		if len(wire.Content) == 0 {
			object["content"] = json.RawMessage(`[]`)
		}
	case "function_call":
		arguments, err := json.Marshal(wire.Arguments)
		if err != nil {
			return nil, err
		}
		object["arguments"] = arguments
	case "reasoning":
		if len(wire.Summary) == 0 {
			object["summary"] = json.RawMessage(`[]`)
		}
	case "image_generation_call":
		if wire.Result == nil {
			object["result"] = json.RawMessage(`null`)
		}
	}
	return json.Marshal(object)
}

type responsesContentWire struct {
	Type                  string                     `json:"type"`
	Text                  string                     `json:"text,omitempty"`
	Refusal               string                     `json:"refusal,omitempty"`
	Annotations           *[]responsesAnnotationWire `json:"annotations,omitempty"`
	ImageURL              string                     `json:"image_url,omitempty"`
	FileURL               string                     `json:"file_url,omitempty"`
	FileID                string                     `json:"file_id,omitempty"`
	FileData              string                     `json:"file_data,omitempty"`
	Filename              string                     `json:"filename,omitempty"`
	Detail                string                     `json:"detail,omitempty"`
	Logprobs              json.RawMessage            `json:"logprobs,omitempty"`
	PromptCacheBreakpoint json.RawMessage            `json:"prompt_cache_breakpoint,omitempty"`
}

func (wire responsesContentWire) MarshalJSON() ([]byte, error) {
	type contentAlias responsesContentWire
	body, err := json.Marshal(contentAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "input_text", "output_text", "reasoning_text", "summary_text":
		object["text"], _ = json.Marshal(wire.Text)
	case "refusal":
		object["refusal"], _ = json.Marshal(wire.Refusal)
	}
	if wire.Type == "output_text" && wire.Annotations == nil {
		object["annotations"] = json.RawMessage(`[]`)
	}
	return json.Marshal(object)
}

type responsesAnnotationWire struct {
	Type       string `json:"type"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex int64  `json:"start_index"`
	EndIndex   int64  `json:"end_index"`
}

func (OpenAIResponsesCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire responsesRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"background": wire.Background, "context_management": wire.ContextManagement,
		"include": wire.Include, "max_tool_calls": wire.MaxToolCalls, "moderation": wire.Moderation,
		"prompt": wire.Prompt, "prompt_cache_key": wire.PromptCacheKey,
		"prompt_cache_retention": wire.PromptCacheRetention,
		"prompt_cache_options":   wire.PromptCacheOptions, "safety_identifier": wire.SafetyIdentifier,
		"service_tier": wire.ServiceTier,
		"top_logprobs": wire.TopLogprobs,
	}); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if wire.MaxOutputTokens != nil && *wire.MaxOutputTokens < 16 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_responses_max_output_tokens",
			"Responses output token limit must be at least 16",
			nil,
		)
	}
	conversationID, err := decodeResponsesConversation(wire.Conversation)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request := decodeResponsesBaseRequest(wire, conversationID)
	if err := decodeResponsesReasoningRequest(wire.Reasoning, &request); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesInstructions(wire.Instructions, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesInput(wire.Input, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesTools(wire.Tools, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if err := decodeResponsesRequestOptions(wire, &request, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	return request, requestEnvelope(llmprotocol.OpenAIResponsesV1, body, request.Generation, policy), nil, nil
}

func decodeResponsesBaseRequest(wire responsesRequestWire, conversationID string) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: 1, Model: wire.Model, Stream: wire.Stream, Metadata: wire.Metadata,
		EndUserID: wire.User, PreviousResponseID: wire.PreviousResponseID, ConversationID: conversationID,
		Truncation: wire.Truncation,
		Store:      wire.Store, AutoStore: wire.AutoStore, ParallelToolCalls: wire.ParallelToolCalls,
		Sampling: llmprotocol.Sampling{Temperature: wire.Temperature, TopP: wire.TopP, MaxOutputTokens: wire.MaxOutputTokens},
		Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIResponsesV1},
	}
	if wire.StreamOptions != nil {
		request.StreamOptions.IncludeObfuscation = wire.StreamOptions.IncludeObfuscation
	}
	return request
}

func decodeResponsesReasoningRequest(reasoning *responsesReasoningWire, request *llmprotocol.Request) error {
	if reasoning == nil {
		return nil
	}
	request.ReasoningEffort = reasoning.Effort
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"reasoning.mode":             reasoning.Mode,
		"reasoning.summary":          reasoning.Summary,
		"reasoning.context":          reasoning.Context,
		"reasoning.generate_summary": reasoning.GenerateSummary,
	}); err != nil {
		return err
	}
	return nil
}

func decodeResponsesInstructions(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var text string
	if err := decodeWireValue(raw, &text, policy); err != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			"invalid_instructions",
			"Responses instructions must be a string",
			err,
		)
	}
	request.Instructions = []llmprotocol.InstructionBlock{{
		Role:    llmprotocol.RoleDeveloper,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}}
	return nil
}

func decodeResponsesTools(raw json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var toolBodies []json.RawMessage
	if err := decodeWireValue(raw, &toolBodies, policy); err != nil {
		return err
	}
	for _, toolBody := range toolBodies {
		if err := decodeResponsesTool(toolBody, request, policy); err != nil {
			return err
		}
	}
	return nil
}

func decodeResponsesTool(body json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	toolType, err := responsesToolDiscriminator(body)
	if err != nil {
		return err
	}
	if err := validateResponsesToolVariant(body, toolType); err != nil {
		return err
	}
	if toolType == "image_generation" {
		return decodeResponsesImageGenerationTool(body, request, policy)
	}
	if toolType != "function" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "only function tools enter the model protocol", nil)
	}
	return decodeResponsesFunctionTool(body, request, policy)
}

func responsesToolDiscriminator(body json.RawMessage) (string, error) {
	var discriminator struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(body, &discriminator); err != nil || discriminator.Type == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses tool type is required", err)
	}
	return discriminator.Type, nil
}

func decodeResponsesImageGenerationTool(
	body json.RawMessage,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	if request.ImageGeneration != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "duplicate_image_generation_tool", "Responses request declares image generation more than once", nil)
	}
	var tool responsesToolWire
	if err := decodeWireValue(body, &tool, policy); err != nil {
		return err
	}
	request.ImageGeneration = decodeResponsesImageGenerationOptions(tool)
	return nil
}

func decodeResponsesFunctionTool(
	body json.RawMessage,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	var tool responsesToolWire
	if err := decodeWireValue(body, &tool, policy); err != nil {
		return err
	}
	if err := rejectUnsupportedRequestFields(map[string]json.RawMessage{
		"tools.allowed_callers": tool.AllowedCallers,
		"tools.defer_loading":   tool.DeferLoading,
		"tools.output_schema":   tool.OutputSchema,
	}); err != nil {
		return err
	}
	schema := tool.Parameters
	if len(schema) == 0 {
		schema = json.RawMessage(`{"type":"object"}`)
	}
	request.Tools = append(request.Tools, llmprotocol.Tool{
		Name: tool.Name, Description: tool.Description, InputSchema: schema, Strict: tool.Strict,
	})
	return nil
}

func validateResponsesToolVariant(body json.RawMessage, toolType string) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses tool is invalid", err)
	}
	allowed := map[string]struct{}{"type": {}}
	var names []string
	switch toolType {
	case "function":
		names = []string{"name", "description", "parameters", "strict", "allowed_callers", "defer_loading", "output_schema"}
	case "image_generation":
		names = []string{
			"model", "quality", "size", "output_format", "output_compression", "moderation",
			"background", "input_fidelity", "input_image_mask", "partial_images", "action",
		}
	default:
		return nil
	}
	for _, name := range names {
		allowed[name] = struct{}{}
	}
	known := []string{
		"name", "description", "parameters", "strict", "allowed_callers", "defer_loading", "output_schema",
		"model", "quality", "size", "output_format", "output_compression", "moderation",
		"background", "input_fidelity", "input_image_mask", "partial_images", "action",
	}
	for _, name := range known {
		if _, present := fields[name]; !present {
			continue
		}
		if _, ok := allowed[name]; !ok {
			return llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest,
				"invalid_tool_variant",
				"Responses tool includes a field from another union variant: "+name,
				nil,
			)
		}
	}
	return nil
}

func decodeResponsesImageGenerationOptions(tool responsesToolWire) *llmprotocol.ImageGenerationOptions {
	options := &llmprotocol.ImageGenerationOptions{
		Model: tool.Model, Quality: tool.Quality, Size: tool.Size,
		OutputFormat: tool.OutputFormat, OutputCompression: tool.OutputCompression,
		Moderation: tool.Moderation, Background: tool.Background,
		InputFidelity: tool.InputFidelity, PartialImages: tool.PartialImages,
		Action: tool.Action,
	}
	if tool.InputImageMask != nil {
		options.InputImageMask = &llmprotocol.ImageGenerationMask{
			EncodedImage: tool.InputImageMask.ImageURL,
			FileID:       tool.InputImageMask.FileID,
		}
	}
	return options
}

func decodeResponsesRequestOptions(wire responsesRequestWire, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	choice, err := decodeResponsesToolChoice(wire.ToolChoice, policy)
	if err != nil {
		return err
	}
	request.ToolChoice = choice
	if wire.Text != nil {
		if err := rejectUnsupportedRequestField("text.verbosity", wire.Text.Verbosity); err != nil {
			return err
		}
		request.OutputFormat = llmprotocol.OutputFormat{
			Kind: llmprotocol.OutputFormatKind(wire.Text.Format.Type), Name: wire.Text.Format.Name,
			Description: wire.Text.Format.Description, Strict: wire.Text.Format.Strict, Schema: wire.Text.Format.Schema,
		}
		switch request.OutputFormat.Kind {
		case "json_schema":
			request.OutputFormat.Kind = llmprotocol.OutputJSONSchema
		case "json_object":
			request.OutputFormat.Kind = llmprotocol.OutputJSONObject
		case "", "text":
			request.OutputFormat.Kind = llmprotocol.OutputText
		default:
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "Responses output format is unsupported", nil)
		}
	}
	return nil
}

func decodeResponsesConversation(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	var id string
	if json.Unmarshal(raw, &id) == nil {
		return id, nil
	}
	var object struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(raw, &object) != nil || object.ID == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_conversation", "conversation must be an ID or object with an ID", nil)
	}
	return object.ID, nil
}
