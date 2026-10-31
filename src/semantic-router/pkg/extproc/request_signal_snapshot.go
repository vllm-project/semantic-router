package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

// requestSignalSnapshot contains the protocol-neutral facts consumed by
// routing signals. It deliberately excludes wire JSON, provider fields, raw
// media bytes and tool schemas. DecisionMedia holds ephemeral references only,
// omitted from serialization; it is not a replay field.
// Tool-result text is confined to the dedicated Guard
// projection; it is not added to the general semantic signal text.
type requestSignalSnapshot struct {
	Model             string
	Stream            bool
	UserContent       string
	PriorUserMessages []string
	NonUserMessages   []string
	JailbreakInput    *classification.JailbreakInput
	HasAssistantReply bool
	LastUserHasText   bool
	DecisionMedia     []*decisionmedia.Source `json:"-"`
	FirstImageURL     string
	FirstAudio        string
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
