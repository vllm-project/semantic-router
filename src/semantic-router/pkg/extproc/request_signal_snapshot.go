package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

// requestSignalSnapshot contains the protocol-neutral facts consumed by
// routing signals. It deliberately excludes wire JSON, provider fields, raw
// media and tool schemas. Tool-result text is confined to the dedicated Guard
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
	FirstImageURL     string
	FirstAudio        string
	// Turns are the user and assistant messages with text, in order; the
	// last user one is the current turn (PriorTurns). Images are every image
	// the request carries (requestImages); UserVideos are the inline videos
	// of the last user message.
	Turns      []classification.ConversationTurn
	Images     []string
	UserVideos []string
	// ToolResults are the tool messages' texts, in order: the content a
	// tool-continuation turn carries instead of user text.
	ToolResults []string
	// InputModality holds the structural input-modality counts for the
	// input_modality signal family, scoped to user messages.
	InputModality classification.InputModalityFacts
	Metadata      map[string]string

	ContextTokenFloor      int
	ContextTextBytes       int
	ContextEquivalentBytes int
	ContextHasNonText      bool
	ContextReserveTokens   int

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
	LastMessageRole           string
	LastMessageToolResult     bool
	LastMessageFlowToolResult bool
	LastAssistantToolCall     bool
	LastUserAfterToolResult   bool

	RoutingInput routinginput.Outcome
}

// PriorTurns are the user and assistant turns before the current user turn.
func (r *requestSignalSnapshot) PriorTurns() []classification.ConversationTurn {
	for index := len(r.Turns) - 1; index >= 0; index-- {
		if r.Turns[index].Role == string(llmprotocol.RoleUser) {
			return append([]classification.ConversationTurn(nil), r.Turns[:index]...)
		}
	}
	return append([]classification.ConversationTurn(nil), r.Turns...)
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
