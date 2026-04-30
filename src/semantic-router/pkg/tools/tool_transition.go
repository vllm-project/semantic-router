package tools

// ToolTransitionContext summarizes request history facts used by tool retrievers.
type ToolTransitionContext struct {
	// RecentToolNames contains assistant tool-call function names in oldest-to-newest
	// order after applying the configured history window. It is nil when no prior
	// tool calls are present.
	RecentToolNames []string
	// UserMessageCount is the number of user messages observed in the request
	// history. It is 0 when no user message is present.
	UserMessageCount int
	// ToolResultCount counts completed tool result messages observed so far. A
	// single assistant message with three parallel tool calls followed by three
	// tool result messages contributes 3 here.
	ToolResultCount int
	// SelectedDecision is the decision selected at the call site, if selection has
	// already happened. It is empty when transition context is extracted before
	// decision selection.
	SelectedDecision string
	// SelectedCategory is the category selected at the call site, if category
	// selection has already happened. It is empty when extracted earlier.
	SelectedCategory string
}
