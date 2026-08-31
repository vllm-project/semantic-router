package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"

// requestSignalSnapshot contains the protocol-neutral facts consumed by
// routing signals. It deliberately excludes wire JSON, provider fields, raw
// media, tool schemas, and tool-result payloads.
type requestSignalSnapshot struct {
	Model             string
	Stream            bool
	UserContent       string
	PriorUserMessages []string
	NonUserMessages   []string
	HasAssistantReply bool
	FirstImageURL     string
	ImageContentCount int
	// InputModality holds the structural input-modality counts for the
	// input_modality signal family, scoped to user messages.
	InputModality classification.InputModalityFacts
	Metadata      map[string]string

	ContextTokenFloor      int
	ContextTextBytes       int
	ContextEquivalentBytes int
	ContextHasNonText      bool

	HasDeveloperMessage       bool
	UserMessageCount          int
	AssistantMessageCount     int
	SystemMessageCount        int
	ToolMessageCount          int
	ToolDefinitionCount       int
	ToolChoiceRequired        bool
	ToolChoiceNone            bool
	AssistantToolCallCount    int
	ToolResultCount           int
	AssistantToolNames        []string
	LastMessageRole           string
	LastMessageToolResult     bool
	LastMessageFlowToolResult bool
	LastAssistantToolCall     bool
	LastUserAfterToolResult   bool
}

func recordNonUserSignalMessage(result *requestSignalSnapshot, role string, text string) {
	if text == "" {
		return
	}
	result.NonUserMessages = append(result.NonUserMessages, text)
	if role == "assistant" {
		result.HasAssistantReply = true
	}
}
