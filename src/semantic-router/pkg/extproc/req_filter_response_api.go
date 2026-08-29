package extproc

import (
	"context"
	"encoding/json"
	"errors"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

// ResponseAPIFilter owns optional Responses object CRUD. Generation is handled
// by the neutral codec runtime and never depends on this store.
type ResponseAPIFilter struct {
	store   responsestore.ResponseStore
	enabled bool
}

// NewResponseAPIFilter creates a new Response API filter.
func NewResponseAPIFilter(store responsestore.ResponseStore) *ResponseAPIFilter {
	return &ResponseAPIFilter{
		store:   store,
		enabled: store != nil && store.IsEnabled(),
	}
}

// IsEnabled returns whether Response API is enabled.
func (f *ResponseAPIFilter) IsEnabled() bool {
	return f.enabled
}

// ResponseObjectState is request-scoped state for optional Responses object
// persistence. It is not part of generation or protocol translation.
type ResponseObjectState struct {
	// GeneratedResponseID is the Router-owned public identity for this turn.
	// Provider response IDs remain transport metadata and never become object
	// store keys.
	GeneratedResponseID string

	// PreviousResponseID from the request (for conversation chaining)
	PreviousResponseID string

	// ConversationHistory fetched from store
	ConversationHistory []*responseapi.StoredResponse

	// ConversationID is the determined ConversationID for this request.
	// Set during neutral request preparation to ensure consistent tracking.
	// Sources (in priority order):
	//   1. Request's conversation_id field
	//   2. First response in conversation chain (via previous_response_id)
	//   3. Newly generated (for new conversations)
	ConversationID string

	// Input, Instructions, and Metadata are immutable ingress snapshots used for
	// retention. Later routing and plugin mutations must not alter stored input.
	Input        []responseapi.InputItem
	Instructions string
	Metadata     map[string]string
	ShouldStore  bool

	// PersistenceAttempted prevents multiple terminal response paths from
	// retaining the same object twice.
	PersistenceAttempted bool

	// ProviderContextApplied makes conversation materialization idempotent
	// across retries and multi-step execution. Object identifiers and store
	// controls belong to this Router edge, not to the selected model protocol.
	ProviderContextApplied bool
}

// PrepareObjectState derives object identity and optional retained history from
// an already validated neutral request. Stateless POST generation remains
// available without a store; an explicit previous_response_id fails closed if
// its history cannot be retrieved.
func (f *ResponseAPIFilter) PrepareObjectState(
	ctx context.Context,
	request llmprotocol.Request,
	sourceBody []byte,
) (*ResponseObjectState, error) {
	state := &ResponseObjectState{
		GeneratedResponseID: responseapi.GenerateResponseID(),
		PreviousResponseID:  request.PreviousResponseID,
		Metadata:            cloneResponseMetadata(request.Metadata),
		ShouldStore:         request.Store == nil || *request.Store,
	}
	state.Input, state.Instructions = snapshotResponseObjectRequest(sourceBody, request)
	if request.PreviousResponseID != "" {
		if f == nil || !f.enabled {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"response_history_unavailable",
				"retained response history is unavailable",
				nil,
			)
		}
		history, err := f.store.GetConversationChain(ctx, request.PreviousResponseID)
		if errors.Is(err, responsestore.ErrNotFound) || (err == nil && len(history) == 0) {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorNotFound,
				"previous_response_not_found",
				"previous response was not found",
				err,
			)
		}
		if err != nil {
			logging.Warnf("Failed to fetch response history (error_class=%T)", err)
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"response_history_unavailable",
				"retained response history is unavailable",
				err,
			)
		}
		state.ConversationHistory = history
	}
	state.ConversationID = determineConversationID(request.ConversationID, state.ConversationHistory)
	return state, nil
}

// HandleGetResponse handles GET /v1/responses/{id} requests.
func (f *ResponseAPIFilter) HandleGetResponse(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if f == nil || !f.enabled {
		return responseObjectNotFound(responseID), nil
	}

	// Get response from store
	stored, err := f.store.GetResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			return responseObjectNotFound(responseID), nil
		}
		logging.Errorf("Response API: Error getting response %s: %v", responseID, err)
		return createResponseAPIError(500, "Error retrieving response"), nil
	}

	// Convert to Response API format
	resp := f.storedToResponseAPIResponse(stored)

	// Marshal response
	body, err := json.Marshal(resp)
	if err != nil {
		logging.Errorf("Response API: Error marshaling response: %v", err)
		return createResponseAPIError(500, "Error serializing response"), nil
	}

	return createImmediateJSONResponse(200, body), nil
}

// HandleDeleteResponse handles DELETE /v1/responses/{id} requests.
func (f *ResponseAPIFilter) HandleDeleteResponse(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if f == nil || !f.enabled {
		return responseObjectNotFound(responseID), nil
	}

	// Delete response from store
	err := f.store.DeleteResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			return responseObjectNotFound(responseID), nil
		}
		logging.Errorf("Response API: Error deleting response %s: %v", responseID, err)
		return createResponseAPIError(500, "Error deleting response"), nil
	}

	// Return deletion confirmation
	deleteResp := responseapi.DeleteResponseResult{
		ID:      responseID,
		Object:  "response.deleted",
		Deleted: true,
	}

	body, err := json.Marshal(deleteResp)
	if err != nil {
		logging.Errorf("Response API: Error marshaling delete response: %v", err)
		return createResponseAPIError(500, "Error serializing response"), nil
	}

	return createImmediateJSONResponse(200, body), nil
}

// HandleGetInputItems handles GET /v1/responses/{id}/input_items requests.
func (f *ResponseAPIFilter) HandleGetInputItems(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if f == nil || !f.enabled {
		return responseObjectNotFound(responseID), nil
	}

	// Get response from store
	stored, err := f.store.GetResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			return responseObjectNotFound(responseID), nil
		}
		logging.Errorf("Response API: Error getting response %s: %v", responseID, err)
		return createResponseAPIError(500, "Error retrieving response"), nil
	}

	// Build input items list from stored response
	inputItems := f.buildInputItemsList(stored)

	// Create response with pagination structure
	listResp := responseapi.InputItemsListResponse{
		Object:  "list",
		Data:    inputItems,
		FirstID: "",
		LastID:  "",
		HasMore: false,
	}

	if len(inputItems) > 0 {
		listResp.FirstID = inputItems[0].ID
		listResp.LastID = inputItems[len(inputItems)-1].ID
	}

	body, err := json.Marshal(listResp)
	if err != nil {
		logging.Errorf("Response API: Error marshaling input items: %v", err)
		return createResponseAPIError(500, "Error serializing response"), nil
	}

	return createImmediateJSONResponse(200, body), nil
}

func responseObjectNotFound(responseID string) *ext_proc.ProcessingResponse {
	return createResponseAPIError(404, "Response not found: "+responseID)
}

func snapshotResponseObjectRequest(sourceBody []byte, request llmprotocol.Request) ([]responseapi.InputItem, string) {
	var wire struct {
		Input        json.RawMessage `json:"input"`
		Instructions json.RawMessage `json:"instructions"`
	}
	if json.Unmarshal(sourceBody, &wire) == nil {
		items := parseResponseAPIInputItems(wire.Input)
		var instructions string
		if json.Unmarshal(wire.Instructions, &instructions) == nil {
			return items, instructions
		}
		return items, responseObjectInstructionText(request.Instructions)
	}
	return nil, responseObjectInstructionText(request.Instructions)
}

func parseResponseAPIInputItems(input json.RawMessage) []responseapi.InputItem {
	if len(input) == 0 {
		return nil
	}
	var items []responseapi.InputItem
	if json.Unmarshal(input, &items) == nil {
		for index := range items {
			if items[index].ID == "" {
				items[index].ID = responseapi.GenerateItemID()
			}
			if items[index].Status == "" {
				items[index].Status = responseapi.StatusCompleted
			}
			if items[index].Type == "" {
				items[index].Type = responseapi.ItemTypeMessage
			}
		}
		return items
	}
	var text string
	if json.Unmarshal(input, &text) == nil {
		return []responseapi.InputItem{{
			ID: responseapi.GenerateItemID(), Type: responseapi.ItemTypeMessage,
			Role: responseapi.RoleUser, Content: append(json.RawMessage(nil), input...),
			Status: responseapi.StatusCompleted,
		}}
	}
	return nil
}

func responseObjectInstructionText(blocks []llmprotocol.InstructionBlock) string {
	var text string
	for _, block := range blocks {
		for _, content := range block.Content {
			if content.Kind != llmprotocol.ContentText || content.Text == "" {
				continue
			}
			if text != "" {
				text += "\n"
			}
			text += content.Text
		}
	}
	return text
}

func cloneResponseMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(metadata))
	for key, value := range metadata {
		cloned[key] = value
	}
	return cloned
}

// buildInputItemsList builds the input items list from a stored response.
func (f *ResponseAPIFilter) buildInputItemsList(stored *responseapi.StoredResponse) []responseapi.InputItem {
	var items []responseapi.InputItem

	// Add instructions as system message if present
	if stored.Instructions != "" {
		contentParts := []responseapi.ContentPart{{Type: "input_text", Text: stored.Instructions}}
		contentJSON, _ := json.Marshal(contentParts)
		items = append(items, responseapi.InputItem{
			ID:      responseapi.GenerateItemID(),
			Type:    "message",
			Role:    "system",
			Content: contentJSON,
			Status:  "completed",
		})
	}

	// Add stored input items
	items = append(items, stored.Input...)

	return items
}

// storedToResponseAPIResponse converts a StoredResponse back to ResponseAPIResponse.
func (f *ResponseAPIFilter) storedToResponseAPIResponse(stored *responseapi.StoredResponse) *responseapi.ResponseAPIResponse {
	return &responseapi.ResponseAPIResponse{
		ID:                 stored.ID,
		Object:             "response",
		CreatedAt:          stored.CreatedAt,
		Model:              stored.Model,
		Status:             stored.Status,
		Output:             stored.Output,
		OutputText:         stored.OutputText,
		PreviousResponseID: stored.PreviousResponseID,
		ConversationID:     stored.ConversationID,
		Usage:              stored.Usage,
		Instructions:       stored.Instructions,
		Metadata:           stored.Metadata,
	}
}

// determineConversationID determines the ConversationID for this request.
// Priority order:
//  1. Request's conversation_id field (explicit)
//  2. First response in conversation chain (continuation via previous_response_id)
//  3. Newly generated (new conversation)
func determineConversationID(explicit string, history []*responseapi.StoredResponse) string {
	// Priority 1: Request explicitly provides conversation_id
	if explicit != "" {
		return explicit
	}

	// Priority 2: Get from conversation history (continuation)
	if len(history) > 0 {
		// Find ConversationID from the first response in the chain
		firstResponse := history[0]
		if firstResponse.ConversationID != "" {
			return firstResponse.ConversationID
		}
	}

	// Priority 3: Generate new ConversationID (new conversation)
	return responseapi.GenerateConversationID()
}

// createResponseAPIError creates an error response in OpenAI format.
func createResponseAPIError(statusCode int, message string) *ext_proc.ProcessingResponse {
	errorResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	}

	body, _ := json.Marshal(errorResp)
	return createImmediateJSONResponse(statusCode, body)
}

// createImmediateJSONResponse creates an immediate response with JSON body.
func createImmediateJSONResponse(statusCode int, body []byte) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToEnumForResponseAPI(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
				},
				Body: body,
			},
		},
	}
}

// statusCodeToEnumForResponseAPI converts HTTP status code to Envoy enum.
func statusCodeToEnumForResponseAPI(statusCode int) typev3.StatusCode {
	switch statusCode {
	case 200:
		return typev3.StatusCode_OK
	case 400:
		return typev3.StatusCode_BadRequest
	case 404:
		return typev3.StatusCode_NotFound
	case 500:
		return typev3.StatusCode_InternalServerError
	default:
		return typev3.StatusCode_OK
	}
}
