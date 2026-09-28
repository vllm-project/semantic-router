package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// materializeResponseObjectContext converts Router-owned response-object state
// into the stateless conversation that a model backend consumes. The Router
// retains object IDs and persistence controls; provider codecs only receive
// model semantics. This keeps one behavior across OpenAI, Anthropic, and future
// provider formats instead of teaching each codec about Router storage.
func (r *OpenAIRouter) materializeResponseObjectContext(request *llmprotocol.Request, ctx *RequestContext) (bool, error) {
	if request == nil || ctx == nil || ctx.ResponseObjectState == nil {
		return false, nil
	}
	state := ctx.ResponseObjectState
	if state.ProviderContextApplied {
		return false, nil
	}

	changed, err := r.prependStoredResponseHistory(request, state.ConversationHistory)
	if err != nil {
		return false, err
	}
	changed = clearResponseObjectControls(request) || changed
	// Tool results decoded from the current request may initially point to a
	// call retained behind previous_response_id. Once the complete history is
	// local, recompute those links so validation proves the actual call/result
	// ordering instead of relying on the storage reference.
	llmprotocol.MarkDeferredToolLinks(request)
	state.ProviderContextApplied = true
	return changed, nil
}

func (r *OpenAIRouter) prependStoredResponseHistory(
	request *llmprotocol.Request,
	stored []*responseapi.StoredResponse,
) (bool, error) {
	if len(stored) == 0 {
		return false, nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return false, err
	}
	history, err := materializeStoredResponseHistory(engine, stored)
	if err != nil {
		return false, fmt.Errorf("materialize retained Responses history: %w", err)
	}
	if len(history) == 0 {
		return false, nil
	}
	request.Messages = append(history, request.Messages...)
	return true, nil
}

func clearResponseObjectControls(request *llmprotocol.Request) bool {
	changed := request.PreviousResponseID != "" || request.ConversationID != "" || request.Store != nil || request.AutoStore != nil
	request.PreviousResponseID = ""
	request.ConversationID = ""
	request.Store = nil
	request.AutoStore = nil
	return changed
}

// materializeStoredResponseHistory decodes retained Responses input and output
// items through the same public codec used at ingress. Storage is not a second
// inference protocol implementation: it only removes resource-lifecycle fields
// before replaying the complete item sequence into the neutral contract.
func materializeStoredResponseHistory(
	engine *protocolcodec.Engine,
	storedResponses []*responseapi.StoredResponse,
) ([]llmprotocol.Message, error) {
	if engine == nil {
		return nil, fmt.Errorf("protocol engine is unavailable")
	}
	items, err := storedResponseItems(storedResponses)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(struct {
		Model string            `json:"model"`
		Input []json.RawMessage `json:"input"`
	}{Model: "retained-response-history", Input: items})
	if err != nil {
		return nil, err
	}
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		return nil, err
	}
	return request.Messages, nil
}

func storedResponseItems(storedResponses []*responseapi.StoredResponse) ([]json.RawMessage, error) {
	items := make([]json.RawMessage, 0)
	for _, stored := range storedResponses {
		if stored == nil {
			continue
		}
		input, err := marshalStoredInputItems(stored.Input)
		if err != nil {
			return nil, err
		}
		output, err := marshalStoredOutputItems(stored.Output)
		if err != nil {
			return nil, err
		}
		items = append(items, input...)
		items = append(items, output...)
		if len(stored.Output) == 0 && stored.OutputText != "" {
			body, err := marshalStoredOutputText(stored.OutputText)
			if err != nil {
				return nil, err
			}
			items = append(items, body)
		}
	}
	return items, nil
}

func marshalStoredInputItems(items []responseapi.InputItem) ([]json.RawMessage, error) {
	encoded := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		item.Status = ""
		body, err := json.Marshal(item)
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, body)
	}
	return encoded, nil
}

func marshalStoredOutputItems(items []responseapi.OutputItem) ([]json.RawMessage, error) {
	encoded := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		item.Status = ""
		body, err := json.Marshal(item)
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, body)
	}
	return encoded, nil
}

func marshalStoredOutputText(text string) (json.RawMessage, error) {
	return json.Marshal(responseapi.OutputItem{
		Type: responseapi.ItemTypeMessage,
		Role: responseapi.RoleAssistant,
		Content: []responseapi.ContentPart{{
			Type: responseapi.ContentTypeOutputText,
			Text: text,
		}},
	})
}
