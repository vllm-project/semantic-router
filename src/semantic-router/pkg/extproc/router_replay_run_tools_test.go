package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func replayToolStep(source, kind, name, id, arguments, status string) routerreplay.ToolTraceStep {
	return routerreplay.ToolTraceStep{
		Type: kind, Source: source, ToolName: name, ToolCallID: id, Arguments: arguments, Status: status,
	}
}

// A Claude Code run kept with tool traces: the main agent's first turn asks
// for two subagents at once and a shell command, the subagents run in
// parallel and the second fails, and the main agent's next request carries
// the three results before it reads a file.
func TestReplayRunShowsToolCallsAndSubagentReturns(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(30, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 17, 9, 0, 0, 0, time.UTC)
	call := func(id, agent string, at int, steps ...routerreplay.ToolTraceStep) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, RequestID: "req-" + id, SessionID: "cc-tools", Timestamp: start.Add(time.Duration(at) * time.Second),
			DurationMS: 1000, OriginalModel: "vllm-sr/auto", SelectedModel: "fable",
			Agent:          &store.AgentIdentity{Harness: store.HarnessClaudeCode, ID: agent},
			LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			ToolTrace: &routerreplay.ToolTrace{Steps: steps},
		}
	}
	response, request := replayToolSourceResponse, replayToolSourceRequest
	asked, result := replayToolStepAssistantToolCall, replayToolStepClientToolResult
	explore := `{"subagent_type":"Explore","description":"Find callers","prompt":"Find every caller of add"}`
	plan := `{"subagent_type":"Plan","description":"Plan the fix"}`
	bash := `{"command":"python3 -m pytest -q"}`
	failedPlan := call("plan-2", "plan", 4)
	failedPlan.LifecycleState, failedPlan.ResponseStatus = routerreplay.LifecycleFailed, 500
	offered := func(record routerreplay.RoutingRecord, tools ...string) routerreplay.RoutingRecord {
		record.ToolTrace.OfferedTools = tools
		return record
	}
	ids := make(map[string]string)
	for _, record := range []routerreplay.RoutingRecord{
		offered(call("main-1", "", 0,
			replayToolStep(response, asked, "Agent", "a1", explore, ""),
			replayToolStep(response, asked, "Agent", "a2", plan, ""),
			replayToolStep(response, asked, "Bash", "b1", bash, "")), "Agent", "Bash", "Read", "ToolSearch"),
		call("explore-1", "explore", 2), call("plan-1", "plan", 3), call("explore-2", "explore", 5), failedPlan,
		// ToolSearch loaded a deferred MCP tool, and the turn no longer offers ToolSearch
		offered(call("main-2", "", 10,
			replayToolStep(request, asked, "Agent", "a1", explore, ""),
			replayToolStep(request, asked, "Agent", "a2", plan, ""),
			replayToolStep(request, asked, "Bash", "b1", bash, ""),
			replayToolStep(request, result, "", "a1", "", "succeeded"),
			replayToolStep(request, result, "", "a2", "", "failed"),
			replayToolStep(request, result, "", "b1", "", "failed"),
			replayToolStep(response, asked, "Read", "r1", `{"file_path":"/work/calc.py"}`, "")), "Agent", "Bash", "Read", "mcp__docs__search"),
		call("main-3", "", 12,
			replayToolStep(request, asked, "Read", "r1", `{"file_path":"/work/calc.py"}`, ""),
			replayToolStep(request, result, "", "r1", "", "succeeded")),
	} {
		id, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
		ids[record.ID] = id
	}

	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-tools", &run))
	byID := make(map[string]routerreplay.RunCall)
	for _, entry := range run.Calls {
		byID[entry.RecordID] = entry
	}
	// The gateway times a turn's tool calls from the end of its response to
	// the start of the next request that carried their results: one window
	// for the calls of one response.
	for id, want := range map[string]int64{"main-1": 9000, "main-2": 1000} {
		for _, tool := range byID[ids[id]].ToolCalls {
			require.Equal(t, "gateway", tool.TimingSource)
			require.NotNil(t, tool.ClientMS, "%s %s", id, tool.Name)
			require.Equal(t, want, *tool.ClientMS, "%s %s", id, tool.Name)
			require.Equal(t, tool.RequestedAt.Add(time.Duration(want)*time.Millisecond), *tool.ReturnedAt)
			require.Nil(t, tool.StartedAt, "no plugin reported this call")
		}
	}
	for id, entry := range byID {
		for index := range entry.ToolCalls {
			tool := &entry.ToolCalls[index]
			tool.TimingSource, tool.RequestedAt, tool.ReturnedAt, tool.ClientMS = "", nil, nil, nil
		}
		byID[id] = entry
	}
	require.Equal(t, []string{"mcp__docs__search"}, byID[ids["main-2"]].OfferedToolsAdded)
	require.Equal(t, []string{"ToolSearch"}, byID[ids["main-2"]].OfferedToolsRemoved)
	require.Empty(t, byID[ids["main-3"]].OfferedToolsAdded, "a turn that offered nothing Replay saw changes nothing")
	require.Equal(t, []routerreplay.RunToolCall{
		{Name: "Agent", ID: "a1", Summary: "Explore · Find callers", Arguments: explore, Status: "succeeded"},
		{Name: "Agent", ID: "a2", Summary: "Plan · Plan the fix", Arguments: plan, Status: "failed"},
		{Name: "Bash", ID: "b1", Summary: "python3 -m pytest -q", Arguments: bash, Status: "failed"},
	}, byID[ids["main-1"]].ToolCalls, "a turn's tool calls, with the results the next request carried")
	require.Equal(t, []routerreplay.RunToolCall{
		{Name: "Read", ID: "r1", Summary: "calc.py", Arguments: `{"file_path":"/work/calc.py"}`, Status: "succeeded"},
	}, byID[ids["main-2"]].ToolCalls)
	require.Empty(t, byID[ids["main-3"]].ToolCalls)

	agents := make(map[string]routerreplay.RunAgent)
	for _, agent := range run.Run.Agents {
		agents[agent.Label] = agent
	}
	explorer, planner := agents["subagent 1"], agents["subagent 2"]
	require.Equal(t, []string{"Agent", "Bash", "Read", "ToolSearch"}, agents["main"].OfferedTools)
	require.Equal(t, ids["main-1"], explorer.SpawnedAfterRecordID)
	require.Equal(t, "Explore · Find callers", explorer.Task, "the first spawn tool call asked for the first subagent")
	require.Equal(t, "Plan · Plan the fix", planner.Task)
	require.Equal(t, ids["main-2"], explorer.ReturnedToRecordID, "both parallel subagents return to the call after the later one ends")
	require.Equal(t, ids["main-2"], planner.ReturnedToRecordID)
	require.Equal(t, "succeeded", explorer.ReturnStatus)
	require.Equal(t, "failed", planner.ReturnStatus)
}

// A turn's final answer is followed by the user's next prompt, whose history
// still ends with the turn's last tool calls and their results. Those calls
// stay with the call that asked for them and never show again under the
// final answer: in Claude Code's shape, where the asking call's response was
// kept, and in Codex's, where the run starts after it.
func TestReplayRunKeepsAToolCallOffTheFinalAnswer(t *testing.T) {
	start := time.Date(2026, 11, 17, 9, 45, 0, 0, time.UTC)
	response, request, answer := replayToolSourceResponse, replayToolSourceRequest, replayToolStepAssistantFinalResponse
	asked, result := replayToolStepAssistantToolCall, replayToolStepClientToolResult
	for _, test := range []struct{ name, harness, tool, id string }{
		{"claude code", store.HarnessClaudeCode, "Bash", "toolu_1"},
		{"codex", store.HarnessCodex, "exec_command", "call_1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			recorder.SetCapturePolicy(true, true, 0)
			router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
			call := func(id string, at, turn int, steps ...routerreplay.ToolTraceStep) routerreplay.RoutingRecord {
				return routerreplay.RoutingRecord{
					ID: id, RequestID: "req-" + id, SessionID: "final-" + test.harness, Timestamp: start.Add(time.Duration(at) * time.Second),
					DurationMS: 1000, OriginalModel: "vllm-sr/auto", SelectedModel: "fable", TurnIndex: turn,
					Agent:          &store.AgentIdentity{Harness: test.harness},
					LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
					ToolTrace: &routerreplay.ToolTrace{Steps: steps},
				}
			}
			history := []routerreplay.ToolTraceStep{
				replayToolStep(request, asked, test.tool, test.id, "", ""),
				replayToolStep(request, result, "", test.id, "", "succeeded"),
			}
			records := []routerreplay.RoutingRecord{
				call("final", 10, 0, append(history, replayToolStep(response, answer, "", "", "", ""))...),
				call("next", 40, 1, append(history, replayToolStep(response, answer, "", "", "", ""))...),
			}
			if test.harness == store.HarnessClaudeCode {
				records = append([]routerreplay.RoutingRecord{call("ask", 0, 0, replayToolStep(response, asked, test.tool, test.id, "", ""))}, records...)
			}
			ids := make(map[string]string)
			for _, record := range records {
				id, err := router.ReplayRecorder.AddRecord(record)
				require.NoError(t, err)
				ids[record.ID] = id
			}
			var run routerreplay.RunResponse
			require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=final-"+test.harness, &run))
			tools := make(map[string][]string)
			for _, entry := range run.Calls {
				for _, tool := range entry.ToolCalls {
					tools[entry.RecordID] = append(tools[entry.RecordID], tool.Name+" "+tool.Status)
				}
			}
			require.Empty(t, tools[ids["final"]], "the final answer asked for no tool")
			require.Empty(t, tools[ids["next"]])
			if test.harness == store.HarnessClaudeCode {
				require.Equal(t, []string{test.tool + " succeeded"}, tools[ids["ask"]])
			}
		})
	}
}

// A Claude Code run whose first turn ran two shell commands, one failed: the
// request that sent their results counts them. Every later request carries
// them again, the main agent's next prompt with no step for the answer
// before it, and so do the auxiliary and compaction requests that copy the
// history and a Looper hop. The run counts two results and one failure, on
// one call, in its header and in Runs.
func TestReplayRunCountsEachToolResultOnce(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(30, 0))
	recorder.SetCapturePolicy(true, true, 0)
	router := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: recorder}
	start := time.Date(2026, 11, 17, 10, 52, 0, 0, time.UTC)
	response, request := replayToolSourceResponse, replayToolSourceRequest
	asked, result, answer := replayToolStepAssistantToolCall, replayToolStepClientToolResult, replayToolStepAssistantFinalResponse
	call := func(id, kind string, at int, steps ...routerreplay.ToolTraceStep) routerreplay.RoutingRecord {
		return routerreplay.RoutingRecord{
			ID: id, RequestID: "req-" + id, RequestKind: kind, SessionID: "cc-counted", Timestamp: start.Add(time.Duration(at) * time.Second),
			OriginalModel: "vllm-sr/auto", SelectedModel: "fable", Agent: &store.AgentIdentity{Harness: store.HarnessClaudeCode},
			LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200, ToolTrace: &routerreplay.ToolTrace{Steps: steps},
		}
	}
	history := []routerreplay.ToolTraceStep{
		replayToolStep(request, asked, "Bash", "b1", "", ""), replayToolStep(request, asked, "Bash", "b2", "", ""),
		replayToolStep(request, result, "", "b1", "", "succeeded"), replayToolStep(request, result, "", "b2", "", "failed"),
	}
	answered := append(append([]routerreplay.ToolTraceStep{}, history...), replayToolStep(response, answer, "", "", "", ""))
	hop := call("hop", "", 9, answered...)
	hop.ParentRequestID = "req-next"
	ids := make(map[string]string)
	for _, record := range []routerreplay.RoutingRecord{
		call("ask", "", 0, replayToolStep(response, asked, "Bash", "b1", "", ""), replayToolStep(response, asked, "Bash", "b2", "", "")),
		call("sent", "", 5, answered...),
		call("title", store.RequestKindAuxiliary, 6, answered...),
		hop, call("next", "", 10, answered...),
		call("compaction", store.RequestKindCompaction, 20, answered...),
	} {
		id, err := router.ReplayRecorder.AddRecord(record)
		require.NoError(t, err)
		ids[record.ID] = id
	}

	var run routerreplay.RunResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-counted", &run))
	require.Equal(t, 2, run.Run.Outcome.ToolResults)
	require.Equal(t, 1, run.Run.Outcome.ToolFailures)
	// One call carried both results first; the later calls that repeat them are no tool loop.
	require.Equal(t, map[string]int{routerreplay.RunMarkerToolLoop: 1, routerreplay.RunMarkerToolFailure: 1}, run.Run.Events)
	counted := make(map[string][2]int)
	for _, entry := range run.Calls {
		counted[entry.RecordID] = [2]int{entry.ToolResults, entry.ToolFailures}
		for _, hop := range entry.Hops {
			counted[hop.RecordID] = [2]int{hop.ToolResults, hop.ToolFailures}
		}
	}
	require.Equal(t, map[string][2]int{
		ids["ask"]: {0, 0}, ids["sent"]: {2, 1}, ids["title"]: {0, 0}, ids["hop"]: {0, 0}, ids["next"]: {0, 0}, ids["compaction"]: {0, 0},
	}, counted)
	var runs routerreplay.RunListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/runs", &runs))
	require.Len(t, runs.Data, 1)
	require.Equal(t, map[string]int{routerreplay.RunMarkerToolLoop: 1, routerreplay.RunMarkerToolFailure: 1}, runs.Data[0].Events)
}

// Without arguments Replay captured, a call reads by its tool's name alone.
func TestReplayToolSummaryReadsTheArgumentAPersonKnows(t *testing.T) {
	for arguments, want := range map[string]string{
		"": "",
		`{"file_path":"/repo/src/calc.py","old_string":"a","new_string":"b"}`: "calc.py",
		`{"cmd":["bash","-lc","pytest -q"]}`:                                  "bash -lc pytest -q",
		`{"agent_type":"worker","message":"Write the tests"}`:                 "worker · Write the tests",
		`{"pattern":"def add"}`:                                               "def add",
		"not json\nat all":                                                    "not json at all",
	} {
		require.Equal(t, want, replayToolSummary(arguments), arguments)
	}
}

// A Codex parent spawns its child in one call and waits for it in the next,
// which is the last call before the child's first: the child is still
// named by the spawn call's message.
func TestReplayRunNamesACodexChildFromAnEarlierSpawnCall(t *testing.T) {
	calls := []routerreplay.RunCall{
		{RecordID: "spawn", Agent: "main", ToolCalls: []routerreplay.RunToolCall{{Name: "multi_agent_v1__spawn_agent", Summary: "List the Python files"}}},
		{RecordID: "wait", Agent: "main", Timestamp: time.Unix(10, 0), ToolCalls: []routerreplay.RunToolCall{{Name: "multi_agent_v1__wait_agent"}}},
		{RecordID: "child-1", Agent: "subagent 1", Timestamp: time.Unix(11, 0), DurationMS: 1000},
		{RecordID: "child-2", Agent: "subagent 1", Timestamp: time.Unix(13, 0), DurationMS: 1000},
		{RecordID: "after", Agent: "main", Timestamp: time.Unix(15, 0)},
	}
	agents := []routerreplay.RunAgent{{Label: "main"}, {Label: "subagent 1", Parent: "main", Kind: "collab_spawn", SpawnedAfterRecordID: "wait"}}
	linkReplayRunAgents(agents, calls)
	require.Equal(t, "List the Python files", agents[1].Task)
	require.Equal(t, "after", agents[1].ReturnedToRecordID)
	require.Equal(t, "succeeded", agents[1].ReturnStatus)
}

// A request's offered tools are kept by name, an MCP or namespaced tool with
// its server, and never with a schema or description.
func TestReplayOfferedToolsKeepNamesOnly(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{Tools: []llmprotocol.Tool{
		{Name: "Bash", Description: "Run a private command", InputSchema: []byte(`{"type":"object"}`)},
		{Name: "mcp__claude__Agent", Description: "Ask Claude Code"},
		{Name: "search", Namespace: "mcp__docs", LocalName: "search"},
	}}}
	trace := buildReplayRequestToolTrace(ctx)
	require.Equal(t, []string{"Bash", "mcp__claude__Agent", "mcp__docs__search"}, trace.OfferedTools)
	structure := routerreplay.ToolTraceStructure(trace)
	require.Equal(t, trace.OfferedTools, structure.OfferedTools)
	require.Empty(t, structure.Steps)
}
