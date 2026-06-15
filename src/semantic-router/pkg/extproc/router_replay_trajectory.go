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
}

type routerReplayTrajectoryResponse struct {
	Object      string              `json:"object"`
	SessionID   string              `json:"session_id"`
	RecordCount int                 `json:"record_count"`
	Messages    []trajectoryMessage `json:"messages"`
}

// handleRouterReplayTrajectoryAPI serves GET /v1/router_replay/trajectory?session_id={id}.
// It converts stored ToolTrace steps into a flat OpenAI Chat Completions message list,
// coalescing consecutive assistant_tool_call steps into a single assistant message.
//
// NOTE: session_id is currently matched against RequestID. A dedicated session index
// will be added once session tracking lands (Issue 1).
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

	payload := routerReplayTrajectoryResponse{
		Object:      "router_replay.trajectory",
		SessionID:   sessionID,
		RecordCount: len(records),
		Messages:    buildTrajectoryMessages(records),
	}
	return r.createRouterReplayJSONResponse(200, payload)
}

// filterTrajectoryRecordsBySession returns records whose RequestID matches sessionID.
func filterTrajectoryRecordsBySession(
	records []routerreplay.RoutingRecord,
	sessionID string,
) []routerreplay.RoutingRecord {
	matched := make([]routerreplay.RoutingRecord, 0)
	for _, record := range records {
		if record.RequestID == sessionID {
			matched = append(matched, record)
		}
	}
	return matched
}

func reverseRoutingRecords(records []routerreplay.RoutingRecord) {
	slices.Reverse(records)
}

// buildTrajectoryMessages converts replay records into an OpenAI-format message list.
// Consecutive assistant_tool_call steps are coalesced into a single assistant message
// with multiple tool_calls, matching OpenAI's expected format.
func buildTrajectoryMessages(records []routerreplay.RoutingRecord) []trajectoryMessage {
	messages := make([]trajectoryMessage, 0)
	var pendingToolCalls []trajectoryToolCall

	flushToolCalls := func() {
		if len(pendingToolCalls) == 0 {
			return
		}
		messages = append(messages, trajectoryMessage{
			Role:      "assistant",
			ToolCalls: pendingToolCalls,
		})
		pendingToolCalls = nil
	}

	for _, record := range records {
		for _, step := range trajectoryStepsForRecord(record) {
			if step.Type == replayToolStepAssistantToolCall {
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
			if msg := trajectoryMessageFromStep(step); msg != nil {
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

func trajectoryMessageFromStep(step routerreplay.ToolTraceStep) *trajectoryMessage {
	switch step.Type {
	case replayToolStepUserInput:
		return &trajectoryMessage{Role: "user", Content: step.Text}
	case replayToolStepClientToolResult:
		return &trajectoryMessage{Role: "tool", Content: step.Text, ToolCallID: step.ToolCallID}
	case replayToolStepAssistantFinalResponse:
		return &trajectoryMessage{Role: "assistant", Content: step.Text}
	default:
		return nil
	}
}
