package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesRequestEchoWire holds the request members a Responses object
// repeats. Each has the same name and shape in the response object.
type responsesRequestEchoWire struct {
	Instructions      json.RawMessage         `json:"instructions,omitempty"`
	Tools             json.RawMessage         `json:"tools,omitempty"`
	ToolChoice        json.RawMessage         `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool                   `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens   *int64                  `json:"max_output_tokens,omitempty"`
	Temperature       *float64                `json:"temperature,omitempty"`
	TopP              *float64                `json:"top_p,omitempty"`
	Text              *responsesTextWire      `json:"text,omitempty"`
	Reasoning         *responsesReasoningWire `json:"reasoning,omitempty"`
	Metadata          map[string]string       `json:"metadata,omitempty"`
	Truncation        string                  `json:"truncation,omitempty"`
	User              string                  `json:"user,omitempty"`
	PromptCacheKey    string                  `json:"prompt_cache_key,omitempty"`
}

// responsesRequestEcho is the request echo of a decoded Responses request, or
// nil when the request sets none of the repeated members. A null member keeps
// the response object's default.
func responsesRequestEcho(wire responsesRequestWire) json.RawMessage {
	echo := responsesRequestEchoWire{
		Instructions: presentJSON(wire.Instructions), Tools: presentJSON(wire.Tools), ToolChoice: presentJSON(wire.ToolChoice),
		ParallelToolCalls: wire.ParallelToolCalls, MaxOutputTokens: wire.MaxOutputTokens,
		Temperature: wire.Temperature, TopP: wire.TopP, Text: wire.Text, Reasoning: wire.Reasoning,
		Metadata: wire.Metadata, Truncation: wire.Truncation, User: wire.User, PromptCacheKey: wire.PromptCacheKey,
	}
	body, err := json.Marshal(echo)
	if err != nil || string(body) == "{}" {
		return nil
	}
	return body
}

func presentJSON(raw json.RawMessage) json.RawMessage {
	if !jsonFieldPresent(raw) {
		return nil
	}
	return raw
}

func newResponsesResponseWire(id, model, status string, createdAt int64, render llmprotocol.ResponseRenderContext) responsesResponseWire {
	wire := responsesResponseWire{
		ID: id, Object: "response", CreatedAt: float64(createdAt), Model: model, Status: status,
		Output:             json.RawMessage(`[]`),
		Metadata:           map[string]string{},
		Instructions:       json.RawMessage(`null`),
		ParallelToolCalls:  json.RawMessage(`true`),
		Temperature:        json.RawMessage(`null`),
		ToolChoice:         json.RawMessage(`"auto"`),
		Tools:              json.RawMessage(`[]`),
		TopP:               json.RawMessage(`null`),
		PreviousResponseID: render.PreviousResponseID,
	}
	if len(render.RequestEcho) > 0 {
		// The echo holds only members of the response object, written by the
		// request codec from a validated request.
		_ = json.Unmarshal(render.RequestEcho, &wire)
	}
	return wire
}
