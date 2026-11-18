package store

// Harnesses an AgentIdentity names.
const (
	HarnessClaudeCode      = "claude_code"
	HarnessCodex           = "codex"
	HarnessOpenCode        = "opencode"
	HarnessDeepSeekHarness = "deepseek_harness"
)

// HarnessAPIClient stands for the records no harness sent, plain API
// clients, in a harness filter and its counts.
const HarnessAPIClient = "api_client"

// FilterHarness is the harness a harness filter matches a record by.
func FilterHarness(record Record) string {
	if record.Agent == nil || record.Agent.Harness == "" {
		return HarnessAPIClient
	}
	return record.Agent.Harness
}

// AgentIdentity places a record in its agent run, the session it is filed
// under: the harness that sent it, the agent that made the call and the agent
// that spawned that one. ID and ParentID are short SHA-256 hashes of the ids
// the harness sent. An empty ID is the run's main agent, and an empty
// ParentID an agent the main agent spawned. Kind is the subagent kind Codex
// names, such as collab_spawn or review.
type AgentIdentity struct {
	Harness  string `json:"harness,omitempty"`
	ID       string `json:"id,omitempty"`
	ParentID string `json:"parent_id,omitempty"`
	Kind     string `json:"kind,omitempty"`
}

func cloneAgentIdentity(value *AgentIdentity) *AgentIdentity {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
