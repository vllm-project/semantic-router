package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/tidwall/sjson"
)

// StreamState tracks Anthropic SSE events while translating to OpenAI SSE.
type StreamState struct {
	MessageID           string
	Created             int64
	RoleSent            bool
	BlockIndexToToolIdx map[int64]int
	NextToolIdx         int
}

// NewStreamState returns initialized stream translation state.
func NewStreamState() *StreamState {
	return &StreamState{
		BlockIndexToToolIdx: make(map[int64]int),
	}
}

// TransformSSEChunkToOpenAI converts Anthropic SSE bytes from Envoy into OpenAI SSE.
// streamDone is true when message_stop is observed.
func TransformSSEChunkToOpenAI(
	anthropicChunk []byte,
	state *StreamState,
	model string,
) ([]byte, bool, error) {
	if state == nil {
		state = NewStreamState()
	}

	var out bytes.Buffer
	streamDone := false
	for _, data := range extractSSEDataLines(anthropicChunk) {
		var event anthropic.MessageStreamEventUnion
		if err := json.Unmarshal(data, &event); err != nil {
			continue
		}
		chunks, done, err := transformStreamEvent(event, state, model)
		if err != nil {
			return nil, false, err
		}
		if done {
			streamDone = true
		}
		for _, chunk := range chunks {
			out.Write(formatOpenAISSELine(chunk))
		}
	}

	if streamDone {
		out.WriteString("data: [DONE]\n\n")
	}

	return out.Bytes(), streamDone, nil
}

func extractSSEDataLines(chunk []byte) [][]byte {
	var lines [][]byte
	for _, line := range bytes.Split(chunk, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			lines = append(lines, payload)
			continue
		}
		if line[0] == '{' {
			lines = append(lines, line)
		}
	}
	return lines
}

func transformStreamEvent(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	switch event.Type {
	case "message_start":
		return handleMessageStart(event, state, model)
	case "content_block_start":
		return handleContentBlockStart(event, state, model)
	case "content_block_delta":
		return handleContentBlockDelta(event, state, model)
	case "message_delta":
		return handleMessageDelta(event, state, model)
	case "message_stop":
		return nil, true, nil
	case "content_block_stop", "ping", "vertex_event":
		return nil, false, nil
	default:
		return nil, false, nil
	}
}

func handleMessageStart(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	start := event.AsMessageStart()
	if start.Message.ID != "" {
		state.MessageID = start.Message.ID
	}
	if state.Created == 0 {
		state.Created = time.Now().Unix()
	}
	if state.RoleSent {
		return nil, false, nil
	}
	state.RoleSent = true
	chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
		Role: "assistant",
	})
	return [][]byte{chunk}, false, err
}

func handleContentBlockStart(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	start := event.AsContentBlockStart()
	block := start.ContentBlock

	switch block.Type {
	case "text":
		if strings.TrimSpace(block.Text) == "" {
			return nil, false, nil
		}
		chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
			Content: block.Text,
		})
		return [][]byte{chunk}, false, err
	case "tool_use", "server_tool_use":
		toolIdx := state.NextToolIdx
		state.NextToolIdx++
		state.BlockIndexToToolIdx[start.Index] = toolIdx
		chunk, err := buildOpenAIStreamToolChunk(state, model, toolIdx, block.ID, block.Name, "")
		return [][]byte{chunk}, false, err
	default:
		return nil, false, nil
	}
}

func handleContentBlockDelta(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	deltaEvent := event.AsContentBlockDelta()
	delta := deltaEvent.Delta

	switch delta.Type {
	case "text_delta":
		textDelta := delta.AsTextDelta()
		if textDelta.Text == "" {
			return nil, false, nil
		}
		chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
			Content: textDelta.Text,
		})
		return [][]byte{chunk}, false, err
	case "input_json_delta":
		toolIdx, ok := state.BlockIndexToToolIdx[deltaEvent.Index]
		if !ok {
			return nil, false, nil
		}
		jsonDelta := delta.AsInputJSONDelta()
		chunk, err := buildOpenAIStreamToolChunk(state, model, toolIdx, "", "", jsonDelta.PartialJSON)
		return [][]byte{chunk}, false, err
	default:
		return nil, false, nil
	}
}

func handleMessageDelta(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	deltaEvent := event.AsMessageDelta()
	finishReason := mapAnthropicStopReasonToOpenAI(deltaEvent.Delta.StopReason)

	chunk, err := buildOpenAIStreamFinishChunk(state, model, finishReason, deltaEvent.Usage)
	if err != nil {
		return nil, false, err
	}
	if chunk == nil {
		return nil, false, nil
	}
	return [][]byte{chunk}, false, nil
}

func buildOpenAIStreamChunk(
	state *StreamState,
	model string,
	delta openai.ChatCompletionChunkChoiceDelta,
) ([]byte, error) {
	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
	return json.Marshal(chunk)
}

func buildOpenAIStreamToolChunk(
	state *StreamState,
	model string,
	toolIdx int,
	id string,
	name string,
	arguments string,
) ([]byte, error) {
	toolCall := openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: int64(toolIdx),
	}
	if id != "" {
		toolCall.ID = id
	}
	if name != "" {
		toolCall.Type = "function"
		toolCall.Function.Name = name
	}
	if arguments != "" {
		if toolCall.Type == "" {
			toolCall.Type = "function"
		}
		toolCall.Function.Arguments = arguments
	}

	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{toolCall},
			},
		}},
	}
	return json.Marshal(chunk)
}

func buildOpenAIStreamFinishChunk(
	state *StreamState,
	model string,
	finishReason string,
	usage anthropic.MessageDeltaUsage,
) ([]byte, error) {
	if finishReason == "" && usage.InputTokens == 0 && usage.OutputTokens == 0 {
		return nil, nil
	}

	choice := openai.ChatCompletionChunkChoice{
		Index: 0,
		Delta: openai.ChatCompletionChunkChoiceDelta{},
	}
	if finishReason != "" {
		choice.FinishReason = finishReason
	}

	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{choice},
	}
	if usage.InputTokens > 0 || usage.OutputTokens > 0 {
		chunk.Usage = openai.CompletionUsage{
			PromptTokens:     usage.InputTokens,
			CompletionTokens: usage.OutputTokens,
			TotalTokens:      usage.InputTokens + usage.OutputTokens,
		}
	}
	return json.Marshal(chunk)
}

func mapAnthropicStopReasonToOpenAI(reason anthropic.StopReason) string {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return "stop"
	case anthropic.StopReasonMaxTokens:
		return "length"
	case anthropic.StopReasonToolUse:
		return "tool_calls"
	default:
		if reason == "" {
			return ""
		}
		return "stop"
	}
}

func formatOpenAISSELine(chunkJSON []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("data: ")
	buf.Write(chunkJSON)
	buf.WriteString("\n\n")
	return buf.Bytes()
}

// WithStreamingRequestBody adds stream=true to an Anthropic request JSON body.
func WithStreamingRequestBody(body []byte) ([]byte, error) {
	return withStreamFlag(body)
}

// BuildStreamingRequestHeaders returns headers for a streaming Anthropic request.
func BuildStreamingRequestHeaders(apiKey string, bodyLength int, messagesPath string) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, true)
}

// withStreamFlag adds stream=true to an Anthropic request JSON body.
func withStreamFlag(body []byte) ([]byte, error) {
	out, err := sjson.SetBytes(body, "stream", true)
	if err != nil {
		return nil, fmt.Errorf("set stream flag: %w", err)
	}
	return out, nil
}
