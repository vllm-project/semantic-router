package extproc

import (
	"strings"
)

// replayStreamingChoices extracts the choices array from a streaming chat
// completions chunk payload.
func replayStreamingChoices(chunkData map[string]interface{}) []map[string]interface{} {
	rawChoices, ok := chunkData["choices"].([]interface{})
	if !ok || len(rawChoices) == 0 {
		return nil
	}

	choices := make([]map[string]interface{}, 0, len(rawChoices))
	for _, rawChoice := range rawChoices {
		choice, ok := rawChoice.(map[string]interface{})
		if ok {
			choices = append(choices, choice)
		}
	}
	return choices
}

// replayStreamingToolCalls pulls the tool-call fragments out of a single
// streaming choice's delta block.
func replayStreamingToolCalls(choice map[string]interface{}) []replayStreamingIndexedToolCall {
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return nil
	}
	rawToolCalls, ok := delta["tool_calls"].([]interface{})
	if !ok {
		return nil
	}

	toolCalls := make([]replayStreamingIndexedToolCall, 0, len(rawToolCalls))
	for rawIndex, rawToolCall := range rawToolCalls {
		toolCall, ok := rawToolCall.(map[string]interface{})
		if ok {
			toolCalls = append(toolCalls, replayStreamingIndexedToolCall{
				rawIndex: rawIndex,
				toolCall: toolCall,
			})
		}
	}
	return toolCalls
}

// mergeReplayStreamingToolCall merges one streaming fragment into the
// accumulating StreamingToolCallState keyed by the tool-call index.
func mergeReplayStreamingToolCall(ctx *RequestContext, rawIndex int, toolCall map[string]interface{}) {
	index := replayStreamingToolCallIndex(rawIndex, toolCall)
	state := replayStreamingToolCallState(ctx, index)

	if id, ok := toolCall["id"].(string); ok && id != "" {
		state.ID = mergeReplayStreamingFragment(state.ID, id)
	}
	if fn, ok := toolCall["function"].(map[string]interface{}); ok {
		mergeReplayStreamingFunctionFragment(state, fn)
	}
}

func replayStreamingToolCallIndex(rawIndex int, toolCall map[string]interface{}) int {
	if value, ok := toolCall["index"].(float64); ok {
		return int(value)
	}
	return rawIndex
}

func replayStreamingToolCallState(ctx *RequestContext, index int) *StreamingToolCallState {
	state := ctx.StreamingToolCalls[index]
	if state != nil {
		return state
	}

	state = &StreamingToolCallState{}
	ctx.StreamingToolCalls[index] = state
	return state
}

func mergeReplayStreamingFunctionFragment(state *StreamingToolCallState, fn map[string]interface{}) {
	if name, ok := fn["name"].(string); ok && name != "" {
		state.Name = mergeReplayStreamingFragment(state.Name, name)
	}
	if arguments, ok := fn["arguments"].(string); ok && arguments != "" {
		state.Arguments = mergeReplayStreamingFragment(state.Arguments, arguments)
	}
}

// mergeReplayStreamingFragment is a best-effort merge between a running
// fragment and a new delta.  Streaming providers occasionally re-send either
// the full-so-far string or only the incremental suffix; this helper
// tolerates both shapes.
func mergeReplayStreamingFragment(current string, fragment string) string {
	if fragment == "" {
		return current
	}
	if current == "" {
		return fragment
	}
	if strings.HasPrefix(fragment, current) {
		return fragment
	}
	if strings.HasPrefix(current, fragment) {
		return current
	}
	return current + fragment
}
