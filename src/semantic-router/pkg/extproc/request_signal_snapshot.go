package extproc

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
	Metadata          map[string]string

	ContextTokenFloor      int
	ContextTextBytes       int
	ContextEquivalentBytes int
	ContextHasNonText      bool

	HasDeveloperMessage     bool
	UserMessageCount        int
	AssistantMessageCount   int
	SystemMessageCount      int
	ToolMessageCount        int
	ToolDefinitionCount     int
	AssistantToolCallCount  int
	ToolResultCount         int
	AssistantToolNames      []string
	LastMessageRole         string
	LastMessageToolResult   bool
	LastUserAfterToolResult bool
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
