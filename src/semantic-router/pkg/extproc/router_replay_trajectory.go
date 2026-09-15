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
	ConversationID string               `json:"conversation_id,omitempty"`
	Role           string               `json:"role"`
	Content        string               `json:"content,omitempty"`
	ToolCalls      []trajectoryToolCall `json:"tool_calls,omitempty"`
	ToolCallID     string               `json:"tool_call_id,omitempty"`
	ToolName       string               `json:"tool_name,omitempty"`
	TurnIndex      int                  `json:"turn_index"`
}

type routerReplayTrajectoryResponse struct {
	Object      string              `json:"object"`
	SessionID   string              `json:"session_id"`
	Recipe      string              `json:"recipe"`
	RecordCount int                 `json:"record_count"`
	TurnCount   int                 `json:"turn_count"`
	Messages    []trajectoryMessage `json:"messages"`
	Routes      []trajectoryRoute   `json:"routes"`
}

// handleRouterReplayTrajectoryAPI serves GET /api/v1/observability/replays/trajectory?session_id={id}.
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
	recipe, scoped := values.Get("recipe"), values.Has("recipe")
	if !scoped {
		for index, record := range records {
			if index > 0 && record.Recipe != recipe {
				return r.createErrorResponse(400, "recipe is required when a session spans multiple recipes")
			}
			recipe = record.Recipe
		}
	}
	records = filterTrajectoryRecordsByRecipe(records, recipe)
	// collectRouterReplayRecords returns newest-first; trajectory needs chronological order.
	reverseRoutingRecords(records)
	turns := buildTrajectoryTurns(records)

	payload := routerReplayTrajectoryResponse{
		Object:      "router_replay.trajectory",
		SessionID:   sessionID,
		Recipe:      recipe,
		RecordCount: len(records),
		TurnCount:   len(turns),
		Messages:    buildTrajectoryMessages(turns),
		Routes:      buildTrajectoryRoutes(records),
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
	ConversationID string
	Index          int
	Steps          []routerreplay.ToolTraceStep
}

// buildTrajectoryTurns collapses the cumulative HTTP requests made by an agent
// loop into one complete trace per conversation and user turn. Later tool-loop
// requests include prior calls and results, so the richest snapshot is authoritative.
func buildTrajectoryTurns(records []routerreplay.RoutingRecord) []trajectoryTurn {
	turns := make([]trajectoryTurn, 0)
	type turnKey struct {
		conversationID string
		index          int
	}
	turnByIndex := make(map[turnKey]int)
	for _, record := range records {
		steps := trajectoryStepsForRecord(record)
		if len(steps) == 0 {
			continue
		}

		key := turnKey{conversationID: record.ConversationID, index: record.TurnIndex}
		turnPosition, exists := turnByIndex[key]
		if !exists {
			turnByIndex[key] = len(turns)
			turns = append(turns, trajectoryTurn{
				ConversationID: record.ConversationID,
				Index:          record.TurnIndex,
				Steps:          append([]routerreplay.ToolTraceStep(nil), steps...),
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
	pendingConversationID := ""

	flushToolCalls := func() {
		if len(pendingToolCalls) == 0 {
			return
		}
		messages = append(messages, trajectoryMessage{
			Role:           "assistant",
			ToolCalls:      pendingToolCalls,
			TurnIndex:      pendingTurnIndex,
			ConversationID: pendingConversationID,
		})
		pendingToolCalls = nil
	}

	for _, turn := range turns {
		for _, step := range turn.Steps {
			if step.Type == replayToolStepAssistantToolCall {
				if len(pendingToolCalls) > 0 && (pendingTurnIndex != turn.Index || pendingConversationID != turn.ConversationID) {
					flushToolCalls()
				}
				pendingTurnIndex = turn.Index
				pendingConversationID = turn.ConversationID
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
				msg.ConversationID = turn.ConversationID
				messages = append(messages, *msg)
			}
		}
	}
	flushToolCalls()
	return messages
}

// trajectoryStepsForRecord projects the current user turn from the semantic
// trace. Individual records retain their full input history, but earlier user
// turns must not be emitted again under the current turn's index. With no
// visible user boundary, preserve the available trace without inferring one.
// Stored public wire bodies are never reparsed as an implicit canonical protocol.
func trajectoryStepsForRecord(record routerreplay.RoutingRecord) []routerreplay.ToolTraceStep {
	if record.ToolTrace != nil && len(record.ToolTrace.Steps) > 0 {
		steps := record.ToolTrace.Steps
		for index := len(steps) - 1; index >= 0; index-- {
			if steps[index].Type == replayToolStepUserInput {
				return steps[index:]
			}
		}
		return steps
	}
	return nil
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
