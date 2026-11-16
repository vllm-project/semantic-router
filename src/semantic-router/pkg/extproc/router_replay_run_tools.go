package extproc

import (
	"encoding/json"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	replayToolSummaryRunes   = 60
	replayToolArgumentsRunes = 240

	replayToolTimingGateway = "gateway"
	replayToolTimingPlugin  = "plugin"
)

// replayToolSummaryKeys are the arguments a tool call reads by, in order:
// a command, a file, a search, then what a subagent was asked.
var replayToolSummaryKeys = []string{
	"command", "cmd", "file_path", "path", "notebook_path", "pattern", "url", "query",
	"description", "message", "prompt", "task",
}

// replayTurnTools is the tool calls a record's response asked for.
func replayTurnTools(trace *routerreplay.ToolTrace) []routerreplay.RunToolCall {
	if trace == nil {
		return nil
	}
	var calls []routerreplay.RunToolCall
	for _, step := range trace.Steps {
		if step.Source != replayToolSourceRequest && step.Type == replayToolStepAssistantToolCall {
			calls = append(calls, replayRunToolCall(step))
		}
	}
	return calls
}

// replayCarriedTools is the tool calls of the model's previous turn that a
// request repeats ahead of their results, each with its result's status.
func replayCarriedTools(trace *routerreplay.ToolTrace) []routerreplay.RunToolCall {
	if trace == nil {
		return nil
	}
	end := -1
	for index, step := range trace.Steps {
		if step.Source == replayToolSourceRequest && step.Type == replayToolStepAssistantToolCall {
			end = index
		}
	}
	if end < 0 {
		return nil
	}
	begin := end
	for begin > 0 && trace.Steps[begin-1].Source == replayToolSourceRequest &&
		trace.Steps[begin-1].Type == replayToolStepAssistantToolCall {
		begin--
	}
	status := make(map[string]string)
	for _, step := range trace.Steps[end+1:] {
		if step.Source == replayToolSourceRequest && step.Type == replayToolStepClientToolResult {
			status[step.ToolCallID] = step.Status
		}
	}
	if len(status) == 0 {
		return nil
	}
	calls := make([]routerreplay.RunToolCall, 0, end-begin+1)
	for _, step := range trace.Steps[begin : end+1] {
		call := replayRunToolCall(step)
		call.Status = status[step.ToolCallID]
		calls = append(calls, call)
	}
	return calls
}

func replayRunToolCall(step routerreplay.ToolTraceStep) routerreplay.RunToolCall {
	return routerreplay.RunToolCall{
		Name: step.ToolName, ID: step.ToolCallID,
		Summary:   replayToolSummary(step.Arguments),
		Arguments: boundedOneLine(step.Arguments, replayToolArgumentsRunes),
	}
}

// replayToolSummary reads a call's arguments by the one a person knows it
// by: a subagent's type with its description, a command, or a file's name.
func replayToolSummary(arguments string) string {
	if strings.TrimSpace(arguments) == "" {
		return ""
	}
	var fields map[string]any
	if json.Unmarshal([]byte(arguments), &fields) != nil {
		return boundedOneLine(arguments, replayToolSummaryRunes)
	}
	var parts []string
	for _, key := range []string{"subagent_type", "agent_type"} {
		if kind, ok := fields[key].(string); ok && strings.TrimSpace(kind) != "" {
			parts = append(parts, strings.TrimSpace(kind))
			break
		}
	}
	for _, key := range replayToolSummaryKeys {
		if text := replayArgumentText(key, fields[key]); text != "" {
			parts = append(parts, text)
			break
		}
	}
	return boundedOneLine(strings.Join(parts, " · "), replayToolSummaryRunes)
}

func replayArgumentText(key string, value any) string {
	switch typed := value.(type) {
	case string:
		if strings.HasSuffix(key, "path") {
			return path.Base(strings.TrimSpace(typed))
		}
		return strings.TrimSpace(typed)
	case []any:
		words := make([]string, 0, len(typed))
		for _, item := range typed {
			if word, ok := item.(string); ok {
				words = append(words, word)
			}
		}
		return strings.Join(words, " ")
	}
	return ""
}

func boundedOneLine(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	if runes := []rune(text); len(runes) > limit {
		return string(runes[:limit-1]) + "…"
	}
	return text
}

// linkReplayRunTools gives each call the tool calls its turn made: those
// its response asked for, else those the agent's next request repeats when
// that request continues the user's turn, with the status of each result
// that next request carried. A tool call an earlier call already has is not
// repeated: a new prompt's history still ends with the last turn's calls.
func linkReplayRunTools(entries []replayRunEntry, calls []routerreplay.RunCall) {
	following := make([]int, len(calls))
	next := make(map[string]int)
	for index := len(calls) - 1; index >= 0; index-- {
		following[index] = -1
		if calls[index].RequestKind != "" {
			continue
		}
		if later, ok := next[calls[index].Agent]; ok {
			following[index] = later
		}
		next[calls[index].Agent] = index
	}
	attributed := make(map[string]bool)
	for index := range calls {
		call := &calls[index]
		if call.RequestKind != "" {
			continue
		}
		tools := entries[index].turnTools
		var returned *time.Time
		if later := following[index]; later >= 0 {
			returned = &calls[later].Timestamp
			carried := entries[later].carriedTools
			if len(tools) == 0 && entries[later].turnIndex != entries[index].turnIndex {
				carried = nil
			}
			carried = slices.DeleteFunc(slices.Clone(carried), func(tool routerreplay.RunToolCall) bool {
				return attributed[tool.ID]
			})
			tools = mergeReplayTools(tools, carried)
		}
		for _, tool := range tools {
			if tool.ID != "" {
				attributed[tool.ID] = true
			}
		}
		call.ToolCalls = replayGatewayToolTimes(tools, *call, returned)
	}
}

// replayGatewayToolTimes times a turn's tool calls as the gateway sees them:
// from the end of the response that asked for them to the start of the
// agent's next request, for each call whose result that request carried. The
// calls of one response share the window.
func replayGatewayToolTimes(tools []routerreplay.RunToolCall, call routerreplay.RunCall, returned *time.Time) []routerreplay.RunToolCall {
	if len(tools) == 0 {
		return tools
	}
	requested := call.Timestamp.Add(time.Duration(call.DurationMS) * time.Millisecond)
	timed := make([]routerreplay.RunToolCall, len(tools))
	for index, tool := range tools {
		tool.TimingSource, tool.RequestedAt = replayToolTimingGateway, &requested
		if returned != nil && tool.Status != "" {
			client := max(returned.Sub(requested).Milliseconds(), 0)
			tool.ReturnedAt, tool.ClientMS = returned, &client
		}
		timed[index] = tool
	}
	return timed
}

// replayToolReport is what a harness plugin reports of one tool call.
type replayToolReport struct {
	startedAt, endedAt time.Time
	approvalMS         *int64
}

// joinReplayToolReports is where a harness plugin's reports of its tool
// calls join the gateway's view, by tool call id (Claude Code's tool_use id,
// Codex's call_id): a reported call takes the plugin's times and its
// timing_source says so. No report reaches the builder yet, so every call
// keeps the gateway's times.
func joinReplayToolReports(calls []routerreplay.RunCall, reports map[string]replayToolReport) {
	if len(reports) == 0 {
		return
	}
	for index := range calls {
		for tool := range calls[index].ToolCalls {
			timed := &calls[index].ToolCalls[tool]
			report, ok := reports[timed.ID]
			if !ok || timed.ID == "" {
				continue
			}
			started, ended := report.startedAt, report.endedAt
			timed.TimingSource, timed.StartedAt, timed.EndedAt, timed.ApprovalMS = replayToolTimingPlugin, &started, &ended, report.approvalMS
		}
	}
}

// linkReplayOfferedTools gives each agent the tools its first turn offered,
// and each later turn the tools it offered beyond, or no longer offered
// after, the agent's previous turn.
func linkReplayOfferedTools(agents []routerreplay.RunAgent, entries []replayRunEntry, calls []routerreplay.RunCall) {
	byLabel := make(map[string]*routerreplay.RunAgent, len(agents))
	for index := range agents {
		byLabel[agents[index].Label] = &agents[index]
	}
	previous := make(map[string][]string)
	for index := range calls {
		call := &calls[index]
		offered := entries[index].offered
		if call.RequestKind != "" || len(offered) == 0 {
			continue
		}
		before, seen := previous[call.Agent]
		previous[call.Agent] = offered
		if !seen {
			if agent := byLabel[call.Agent]; agent != nil {
				agent.OfferedTools = offered
			}
			continue
		}
		call.OfferedToolsAdded, call.OfferedToolsRemoved = replayToolsDifference(offered, before), replayToolsDifference(before, offered)
	}
}

// replayToolsDifference is the names in tools that others lacks, in order.
func replayToolsDifference(tools, others []string) []string {
	var difference []string
	for _, name := range tools {
		if !slices.Contains(others, name) {
			difference = append(difference, name)
		}
	}
	return difference
}

func mergeReplayTools(asked, carried []routerreplay.RunToolCall) []routerreplay.RunToolCall {
	if len(asked) == 0 {
		return carried
	}
	byID := make(map[string]routerreplay.RunToolCall, len(carried))
	for _, call := range carried {
		byID[call.ID] = call
	}
	merged := make([]routerreplay.RunToolCall, len(asked))
	for index, call := range asked {
		if repeated, ok := byID[call.ID]; ok && call.ID != "" {
			call.Status = repeated.Status
			if call.Arguments == "" {
				call.Summary, call.Arguments = repeated.Summary, repeated.Arguments
			}
		}
		merged[index] = call
	}
	return merged
}

func isReplaySpawnTool(name string) bool {
	return name == "Agent" || name == "Task" || strings.HasSuffix(name, "spawn_agent")
}

// linkReplayRunAgents says what each subagent was asked and which of its
// parent's calls took its result and how. A parent's spawn tool calls go to
// its subagents in the order both happened: a Claude Code turn asks for its
// subagents at once, and a Codex parent may wait in a later call.
func linkReplayRunAgents(agents []routerreplay.RunAgent, calls []routerreplay.RunCall) {
	turns := make(map[string][]int)
	spawns := make(map[string][]replaySpawn)
	for index, call := range calls {
		if call.RequestKind != "" {
			continue
		}
		turns[call.Agent] = append(turns[call.Agent], index)
		for tool := range call.ToolCalls {
			if isReplaySpawnTool(call.ToolCalls[tool].Name) {
				spawns[call.Agent] = append(spawns[call.Agent], replaySpawn{at: call.Timestamp, tool: &calls[index].ToolCalls[tool]})
			}
		}
	}
	for index := range agents {
		agent := &agents[index]
		own := turns[agent.Label]
		if agent.Parent == "" || len(own) == 0 {
			continue
		}
		var spawn *routerreplay.RunToolCall
		if queue := spawns[agent.Parent]; len(queue) > 0 && !queue[0].at.After(calls[own[0]].Timestamp) {
			spawn, spawns[agent.Parent] = queue[0].tool, queue[1:]
			agent.Task = spawn.Summary
		}
		replayRunReturn(agent, spawn, calls, own[len(own)-1], turns[agent.Parent])
	}
}

type replaySpawn struct {
	at   time.Time
	tool *routerreplay.RunToolCall
}

// replayRunReturn finds the parent's first call after the agent's last one
// ended: the call that took its result.
func replayRunReturn(agent *routerreplay.RunAgent, spawn *routerreplay.RunToolCall, calls []routerreplay.RunCall, lastIndex int, parent []int) {
	last := calls[lastIndex]
	ended := last.Timestamp.Add(time.Duration(last.DurationMS) * time.Millisecond)
	for _, index := range parent {
		if calls[index].Timestamp.Before(ended) {
			continue
		}
		agent.ReturnedToRecordID = calls[index].RecordID
		agent.ReturnStatus = "succeeded"
		if last.LifecycleState == routerreplay.LifecycleFailed || last.ResponseStatus >= 400 ||
			(spawn != nil && spawn.Status == "failed") {
			agent.ReturnStatus = "failed"
		}
		return
	}
}
