package responseapi

// ContentPart represents a content part in a message.
type ContentPart struct {
	// Type is the content type: "input_text", "output_text", "input_image", "input_file", etc.
	Type string `json:"type"`

	// Text is the text content (for text types)
	Text string `json:"text,omitempty"`

	// Annotations for the content
	Annotations []Annotation `json:"annotations,omitempty"`

	// Image fields
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileData string `json:"file_data,omitempty"`
	Filename string `json:"filename,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// Annotation represents an annotation on content.
type Annotation struct {
	Type         string        `json:"type"`
	Text         string        `json:"text,omitempty"`
	StartIdx     int           `json:"start_index,omitempty"`
	EndIdx       int           `json:"end_index,omitempty"`
	FileID       string        `json:"file_id,omitempty"`
	FileCitation *FileCitation `json:"file_citation,omitempty"`
}

// FileCitation represents a file citation annotation.
type FileCitation struct {
	FileID string `json:"file_id"`
	Quote  string `json:"quote,omitempty"`
}

// Tool represents a tool definition.
type Tool struct {
	// Type is the tool type: "function", "code_interpreter", "mcp", "image_generation"
	Type string `json:"type"`

	// Function definition (for function type)
	Function *FunctionDef `json:"function,omitempty"`

	// Name is the tool name (for function type shorthand)
	Name string `json:"name,omitempty"`

	// Description of the tool
	Description string `json:"description,omitempty"`

	// Parameters JSON schema (for function type shorthand)
	Parameters interface{} `json:"parameters,omitempty"`

	// MCP server configuration
	ServerLabel     string            `json:"server_label,omitempty"`
	ServerURL       string            `json:"server_url,omitempty"`
	Headers         map[string]string `json:"headers,omitempty"`
	RequireApproval string            `json:"require_approval,omitempty"`

	// Container configuration (for code_interpreter)
	Container *Container `json:"container,omitempty"`

	// Image generation tool parameters (for image_generation type)
	// These fields match the OpenAI Responses API ImageGeneration tool spec.
	ImageGenModel         string `json:"image_gen_model,omitempty" yaml:"image_gen_model,omitempty"`
	ImageGenQuality       string `json:"image_gen_quality,omitempty" yaml:"image_gen_quality,omitempty"`
	ImageGenSize          string `json:"size,omitempty" yaml:"size,omitempty"`
	ImageGenOutputFormat  string `json:"output_format,omitempty" yaml:"output_format,omitempty"`
	ImageGenBackground    string `json:"background,omitempty" yaml:"background,omitempty"`
	ImageGenAction        string `json:"action,omitempty" yaml:"action,omitempty"`
}

// ExtractImageGenParams extracts ImageGenerationToolParams from a Tool of type "image_generation".
// Returns nil if the tool is not an image_generation tool.
func (t *Tool) ExtractImageGenParams() *ImageGenerationToolParams {
	if t.Type != ToolTypeImageGeneration {
		return nil
	}
	return &ImageGenerationToolParams{
		Model:        t.ImageGenModel,
		Quality:      t.ImageGenQuality,
		Size:         t.ImageGenSize,
		OutputFormat: t.ImageGenOutputFormat,
		Background:   t.ImageGenBackground,
		Action:       t.ImageGenAction,
	}
}

// FunctionDef represents a function definition for tool calling.
type FunctionDef struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`
}

// Container represents a code interpreter container configuration.
type Container struct {
	Type    string   `json:"type"`
	FileIDs []string `json:"file_ids,omitempty"`
}

// Usage represents token usage statistics.
type Usage struct {
	InputTokens         int                  `json:"input_tokens"`
	OutputTokens        int                  `json:"output_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	OutputTokensDetails *OutputTokensDetails `json:"output_tokens_details,omitempty"`
}

// OutputTokensDetails provides detailed breakdown of output tokens.
type OutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// ResponseError represents an error in a response.
type ResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// IncompleteDetails explains why a response is incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// Reasoning contains reasoning configuration and output.
type Reasoning struct {
	Effort           string `json:"effort,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// TextConfig contains text generation configuration.
type TextConfig struct {
	Format interface{} `json:"format,omitempty"`
}

// Truncation contains truncation strategy configuration.
type Truncation struct {
	Type string `json:"type,omitempty"`
}

// ResponseStatus constants
const (
	StatusCompleted  = "completed"
	StatusFailed     = "failed"
	StatusInProgress = "in_progress"
	StatusCancelled  = "cancelled"
	StatusQueued     = "queued"
)

// ItemType constants
const (
	ItemTypeMessage            = "message"
	ItemTypeFunctionCall       = "function_call"
	ItemTypeFunctionCallOutput = "function_call_output"
	ItemTypeItemReference      = "item_reference"
	ItemTypeImageGenerationCall = "image_generation_call"
)

// ToolType constants
const (
	ToolTypeFunction        = "function"
	ToolTypeImageGeneration = "image_generation"
	ToolTypeCodeInterpreter = "code_interpreter"
	ToolTypeMCP             = "mcp"
)

// ImageGenerationToolParams holds parameters extracted from an image_generation
// tool definition in the OpenAI Responses API. These map to the fields on the
// ImageGeneration tool spec (see https://developers.openai.com/api/reference).
type ImageGenerationToolParams struct {
	// Model is the image generation model to use (e.g. "gpt-image-1")
	Model string `json:"model,omitempty"`

	// Quality of the generated image: "low", "medium", "high", or "auto"
	Quality string `json:"quality,omitempty"`

	// Size of the generated image: "1024x1024", "1024x1536", "1536x1024", or "auto"
	Size string `json:"size,omitempty"`

	// OutputFormat of the generated image: "png", "webp", or "jpeg"
	OutputFormat string `json:"output_format,omitempty"`

	// Background type: "transparent", "opaque", or "auto"
	Background string `json:"background,omitempty"`

	// Action: "generate", "edit", or "auto"
	Action string `json:"action,omitempty"`

	// OutputCompression level (0-100)
	OutputCompression *int `json:"output_compression,omitempty"`

	// PartialImages count for streaming (0-3)
	PartialImages *int `json:"partial_images,omitempty"`
}

// ContentType constants
const (
	ContentTypeInputText  = "input_text"
	ContentTypeOutputText = "output_text"
	ContentTypeInputImage = "input_image"
	ContentTypeInputFile  = "input_file"
)

// Role constants
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleSystem    = "system"
)
