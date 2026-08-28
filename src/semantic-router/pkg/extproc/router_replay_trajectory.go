package extproc

import (
	"net/url"
	"slices"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const routerReplayTrajectoryPath = routerReplayAPIBasePath + "/trajectory"

type trajectoryFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type trajectoryToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function trajectoryFunctionCall `json:"function"`
}

type trajectoryMessage struct {
	Role       string               `json:"role"`
	Content    string               `json:"content,omitempty"`
	ToolCalls  []trajectoryToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
	ToolName   string               `json:"tool_name,omitempty"`
	TurnIndex  int                  `json:"turn_index"`
}

type routerReplayTrajectoryResponse struct {
	Object      string              `json:"object"`
	SessionID   string              `json:"session_id"`
	RecordCount int                 `json:"record_count"`
	TurnCount   int                 `json:"turn_count"`
	Messages    []trajectoryMessage `json:"messages"`
}

// handleRouterReplayTrajectoryAPI serves GET /v1/router_replay/trajectory?session_id={id}.
// It converts stored ToolTrace steps into a flat OpenAI Chat Completions message list.
// Multiple HTTP requests made by one agent turn are cumulative snapshots; the most
// complete snapshot is selected before messages are emitted.
func (r *OpenAIRouter) handleRouterReplayTrajectoryAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return r.createErrorResponse(400, "invalid query parameters")
	}

	sessionID := strings.TrimSpace(values.Get("session_id"))
	if sessionID == "" {
		return r.createErrorResponse(400, "session_id is required")
	}

	records := filterTrajectoryRecordsBySession(r.collectRouterReplayRecords(), sessionID)
	// collectRouterReplayRecords returns newest-first; trajectory needs chronological order.
	reverseRoutingRecords(records)
	turns := buildTrajectoryTurns(records)

	payload := routerReplayTrajectoryResponse{
		Object:      "router_replay.trajectory",
		SessionID:   sessionID,
		RecordCount: len(records),
		TurnCount:   len(turns),
		Messages:    buildTrajectoryMessages(turns),
	}
	return r.createRouterReplayJSONResponse(200, payload)
}

// filterTrajectoryRecordsBySession returns records captured for one logical session.
func filterTrajectoryRecordsBySession(
	records []routerreplay.RoutingRecord,
	sessionID string,
) []routerreplay.RoutingRecord {
	matched := make([]routerreplay.RoutingRecord, 0)
	for _, record := range records {
		if record.SessionID == sessionID {
			matched = append(matched, record)
		}
	}
	return matched
}

func reverseRoutingRecords(records []routerreplay.RoutingRecord) {
	slices.Reverse(records)
}

type trajectoryTurn struct {
	Index int
	Steps []routerreplay.ToolTraceStep
}

// buildTrajectoryTurns collapses the cumulative HTTP requests made by an agent
// loop into one complete trace per user turn. Later tool-loop requests include
// the prior call and its result, so the richest snapshot is authoritative.
func buildTrajectoryTurns(records []routerreplay.RoutingRecord) []trajectoryTurn {
	turns := make([]trajectoryTurn, 0)
	turnByIndex := make(map[int]int)
	for _, record := range records {
		steps := trajectoryStepsForRecord(record)
		if len(steps) == 0 {
			continue
		}

		turnPosition, exists := turnByIndex[record.TurnIndex]
		if !exists {
			turnByIndex[record.TurnIndex] = len(turns)
			turns = append(turns, trajectoryTurn{
				Index: record.TurnIndex,
				Steps: append([]routerreplay.ToolTraceStep(nil), steps...),
			})
			continue
		}

		if trajectorySnapshotScore(steps) >= trajectorySnapshotScore(turns[turnPosition].Steps) {
			turns[turnPosition].Steps = append([]routerreplay.ToolTraceStep(nil), steps...)
		}
	}
	return turns
}

func trajectorySnapshotScore(steps []routerreplay.ToolTraceStep) int {
	score := len(steps)
	for _, step := range steps {
		switch step.Type {
		case replayToolStepAssistantFinalResponse:
			score += 10_000
		case replayToolStepClientToolResult:
			score += 100
		case replayToolStepAssistantToolCall:
			score += 10
		}
	}
	return score
}

// buildTrajectoryMessages converts replay records into an OpenAI-format message list.
// Consecutive assistant_tool_call steps are coalesced into a single assistant message
// with multiple tool_calls, matching OpenAI's expected format.
func buildTrajectoryMessages(turns []trajectoryTurn) []trajectoryMessage {
	messages := make([]trajectoryMessage, 0)
	var pendingToolCalls []trajectoryToolCall
	pendingTurnIndex := 0

	flushToolCalls := func() {
		if len(pendingToolCalls) == 0 {
			return
		}
		messages = append(messages, trajectoryMessage{
			Role:      "assistant",
			ToolCalls: pendingToolCalls,
			TurnIndex: pendingTurnIndex,
		})
		pendingToolCalls = nil
	}

	for _, turn := range turns {
		for _, step := range turn.Steps {
			if step.Type == replayToolStepAssistantToolCall {
				if len(pendingToolCalls) > 0 && pendingTurnIndex != turn.Index {
					flushToolCalls()
				}
				pendingTurnIndex = turn.Index
				pendingToolCalls = append(pendingToolCalls, trajectoryToolCall{
					ID:   step.ToolCallID,
					Type: "function",
					Function: trajectoryFunctionCall{
						Name:      step.ToolName,
						Arguments: step.Arguments,
					},
				})
				continue
			}
			flushToolCalls()
			if msg := trajectoryMessageFromStep(step, turn.Index); msg != nil {
				messages = append(messages, *msg)
			}
		}
	}
	flushToolCalls()
	return messages
}

// trajectoryStepsForRecord returns the ToolTraceStep slice for a record.
// If ToolTrace is nil or empty, it falls back to parsing the stored request/response bodies.
func trajectoryStepsForRecord(record routerreplay.RoutingRecord) []routerreplay.ToolTraceStep {
	if record.ToolTrace != nil && len(record.ToolTrace.Steps) > 0 {
		return record.ToolTrace.Steps
	}
	trace := fallbackTrajectoryTrace(record)
	if trace != nil {
		return trace.Steps
	}
	return nil
}

// fallbackTrajectoryTrace parses request_body and response_body as Chat Completions
// payloads when tool_trace is absent.
func fallbackTrajectoryTrace(record routerreplay.RoutingRecord) *routerreplay.ToolTrace {
	requestTrace := parseChatCompletionRequestToolTrace([]byte(record.RequestBody))
	responseTrace := parseChatCompletionResponseToolTrace([]byte(record.ResponseBody))
	return mergeReplayToolTraces(requestTrace, responseTrace)
}

func trajectoryMessageFromStep(step routerreplay.ToolTraceStep, turnIndex int) *trajectoryMessage {
	switch step.Type {
	case replayToolStepUserInput:
		return &trajectoryMessage{Role: "user", Content: step.Text, TurnIndex: turnIndex}
	case replayToolStepClientToolResult:
		return &trajectoryMessage{
			Role:       "tool",
			Content:    step.Text,
			ToolCallID: step.ToolCallID,
			ToolName:   step.ToolName,
			TurnIndex:  turnIndex,
		}
	case replayToolStepAssistantFinalResponse:
		return &trajectoryMessage{Role: "assistant", Content: step.Text, TurnIndex: turnIndex}
	default:
		return nil
	}
}
