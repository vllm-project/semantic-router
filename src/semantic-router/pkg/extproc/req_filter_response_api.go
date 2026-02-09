package extproc

import (
	"context"
	"encoding/json"
	"errors"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

// ResponseAPIFilter handles Response API request/response translation.
// It translates /v1/responses requests to /v1/chat/completions format
// and translates responses back, managing conversation history via previous_response_id.
type ResponseAPIFilter struct {
	store      responsestore.ResponseStore
	translator *responseapi.Translator
	enabled    bool
}

// NewResponseAPIFilter creates a new Response API filter.
func NewResponseAPIFilter(store responsestore.ResponseStore) *ResponseAPIFilter {
	return &ResponseAPIFilter{
		store:      store,
		translator: responseapi.NewTranslator(),
		enabled:    store != nil && store.IsEnabled(),
	}
}

// IsEnabled returns whether Response API is enabled.
func (f *ResponseAPIFilter) IsEnabled() bool {
	return f.enabled
}

// ResponseAPIContext holds context for a Response API request during processing.
type ResponseAPIContext struct {
	// IsResponseAPIRequest indicates this is a /v1/responses request
	IsResponseAPIRequest bool

	// OriginalRequest is the parsed Response API request
	OriginalRequest *responseapi.ResponseAPIRequest

	// PreviousResponseID from the request (for conversation chaining)
	PreviousResponseID string

	// ConversationHistory fetched from store
	ConversationHistory []*responseapi.StoredResponse

	// GeneratedResponseID is the ID generated for this response
	GeneratedResponseID string

	// ConversationID is the determined ConversationID for this request.
	// Set early during TranslateRequest to ensure consistent tracking.
	// Sources (in priority order):
	//   1. Request's conversation_id field
	//   2. First response in conversation chain (via previous_response_id)
	//   3. Newly generated (for new conversations)
	ConversationID string

	// TranslatedBody is the Chat Completions request body after translation
	TranslatedBody []byte
}

// TranslateRequest translates a Response API request to Chat Completions format.
// Returns the translated request body and context, or nil if not a Response API request.
func (f *ResponseAPIFilter) TranslateRequest(ctx context.Context, body []byte) (*ResponseAPIContext, []byte, error) {
	if !f.enabled {
		return nil, nil, nil
	}

	// Parse Response API request
	var req responseapi.ResponseAPIRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, nil // Not a valid Response API request, pass through
	}

	// Check if this looks like a Response API request (has input field)
	if len(req.Input) == 0 {
		return nil, nil, nil // Not a Response API request
	}

	// Create context for this request
	respCtx := &ResponseAPIContext{
		IsResponseAPIRequest: true,
		OriginalRequest:      &req,
		PreviousResponseID:   req.PreviousResponseID,
		GeneratedResponseID:  responseapi.GenerateResponseID(),
	}

	// Fetch conversation history if previous_response_id is provided
	if req.PreviousResponseID != "" {
		history, err := f.store.GetConversationChain(ctx, req.PreviousResponseID)
		if err != nil && !errors.Is(err, responsestore.ErrNotFound) {
			logging.Warnf("Failed to fetch conversation history for %s: %v", req.PreviousResponseID, err)
			// Continue without history - don't fail the request
		}
		respCtx.ConversationHistory = history
		logging.Infof("Response API: Fetched %d messages from conversation history", len(history))
	}

	// Determine ConversationID early for consistent tracking across the request lifecycle.
	// This ensures memory extraction can correctly track turns per conversation.
	respCtx.ConversationID = f.determineConversationID(&req, respCtx.ConversationHistory)
	logging.Infof("Response API: ConversationID=%s (source: %s)",
		respCtx.ConversationID, f.getConversationIDSource(&req, respCtx.ConversationHistory))

	// Translate to Chat Completions request
	completionReq, err := f.translator.TranslateToCompletionRequest(&req, respCtx.ConversationHistory)
	if err != nil {
		return nil, nil, err
	}

	// Marshal translated request
	translatedBody, err := json.Marshal(completionReq)
	if err != nil {
		return nil, nil, err
	}

	// Store translated body in context for later use
	respCtx.TranslatedBody = translatedBody

	logging.Infof("Response API: Translated request (model=%s, previous_id=%s, history_len=%d)",
		req.Model, req.PreviousResponseID, len(respCtx.ConversationHistory))

	return respCtx, translatedBody, nil
}

// TranslateResponse translates a Chat Completions response back to Response API format.
func (f *ResponseAPIFilter) TranslateResponse(ctx context.Context, respCtx *ResponseAPIContext, body []byte) ([]byte, error) {
	if !f.enabled || respCtx == nil || !respCtx.IsResponseAPIRequest {
		return body, nil
	}

	// Check if this is an error response (contains "error" field)
	var errorCheck map[string]interface{}
	if err := json.Unmarshal(body, &errorCheck); err == nil {
		if _, hasError := errorCheck["error"]; hasError {
			logging.Warnf("Response API: Backend returned error response, passing through")
			return body, nil
		}
	}

	// Parse Chat Completions response
	var completionResp responseapi.ChatCompletionResponse
	if err := json.Unmarshal(body, &completionResp); err != nil {
		logging.Errorf("Response API: Failed to parse completion response: %v", err)
		return body, nil // Return original on parse error
	}

	// Validate that we have a valid completion response
	if completionResp.ID == "" && len(completionResp.Choices) == 0 {
		logging.Warnf("Response API: Invalid completion response (no id or choices), passing through")
		return body, nil
	}

	// Translate to Response API format
	responseAPIResp := f.translator.TranslateToResponseAPIResponse(
		respCtx.OriginalRequest,
		&completionResp,
		respCtx.PreviousResponseID,
	)

	// Override ID with pre-generated ID
	responseAPIResp.ID = respCtx.GeneratedResponseID

	// Use the ConversationID determined during request translation.
	// This ensures consistent ConversationID across the request lifecycle,
	// which is critical for memory extraction turn tracking.
	if respCtx.ConversationID != "" {
		responseAPIResp.ConversationID = respCtx.ConversationID
	}

	// Store response if enabled
	shouldStore := respCtx.OriginalRequest.Store == nil || *respCtx.OriginalRequest.Store
	if shouldStore && f.store.IsEnabled() {
		stored := f.toStoredResponse(respCtx.OriginalRequest, responseAPIResp)
		if err := f.store.StoreResponse(ctx, stored); err != nil {
			logging.Warnf("Response API: Failed to store response: %v", err)
		} else {
			logging.Infof("Response API: Stored response %s", responseAPIResp.ID)
		}
	}

	// Marshal Response API response
	translatedBody, err := json.Marshal(responseAPIResp)
	if err != nil {
		logging.Errorf("Response API: Failed to marshal response: %v", err)
		return body, nil
	}

	logging.Infof("Response API: Translated response (id=%s, status=%s)", responseAPIResp.ID, responseAPIResp.Status)
	return translatedBody, nil
}

// toStoredResponse converts request and response to a StoredResponse for storage.
func (f *ResponseAPIFilter) toStoredResponse(req *responseapi.ResponseAPIRequest, resp *responseapi.ResponseAPIResponse) *responseapi.StoredResponse {
	inputItems := parseResponseAPIInputItems(req.Input)

	return &responseapi.StoredResponse{
		ID:                 resp.ID,
		Object:             resp.Object,
		CreatedAt:          resp.CreatedAt,
		Model:              resp.Model,
		Status:             resp.Status,
		Input:              inputItems,
		Output:             resp.Output,
		OutputText:         resp.OutputText,
		PreviousResponseID: resp.PreviousResponseID,
		ConversationID:     resp.ConversationID,
		Usage:              resp.Usage,
		Instructions:       resp.Instructions,
		Metadata:           resp.Metadata,
	}
}

func parseResponseAPIInputItems(input json.RawMessage) []responseapi.InputItem {
	if len(input) == 0 {
		return nil
	}

	// Try parsing as array of input items first.
	var items []responseapi.InputItem
	if err := json.Unmarshal(input, &items); err == nil {
		for i := range items {
			if items[i].ID == "" {
				items[i].ID = responseapi.GenerateItemID()
			}
			if items[i].Status == "" {
				items[i].Status = responseapi.StatusCompleted
			}
		}
		return items
	}

	// Fallback: input can also be a string; store as a user message item.
	var inputStr string
	if err := json.Unmarshal(input, &inputStr); err == nil {
		return []responseapi.InputItem{{
			ID:      responseapi.GenerateItemID(),
			Type:    responseapi.ItemTypeMessage,
			Role:    responseapi.RoleUser,
			Content: input,
			Status:  responseapi.StatusCompleted,
		}}
	}

	return nil
}

// HandleGetResponse handles GET /v1/responses/{id} requests.
func (f *ResponseAPIFilter) HandleGetResponse(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if !f.enabled {
		return createResponseAPIError(404, "Response API not enabled"), nil
	}

	// Get response from store
	stored, err := f.store.GetResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			logging.Infof("Response API: Response not found: %s", responseID)
			return createResponseAPIError(404, "Response not found: "+responseID), nil
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

	logging.Infof("Response API: Retrieved response %s", responseID)
	return createImmediateJSONResponse(200, body), nil
}

// HandleDeleteResponse handles DELETE /v1/responses/{id} requests.
func (f *ResponseAPIFilter) HandleDeleteResponse(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if !f.enabled {
		return createResponseAPIError(404, "Response API not enabled"), nil
	}

	// Delete response from store
	err := f.store.DeleteResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			logging.Infof("Response API: Response not found for deletion: %s", responseID)
			return createResponseAPIError(404, "Response not found: "+responseID), nil
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

	logging.Infof("Response API: Deleted response %s", responseID)
	return createImmediateJSONResponse(200, body), nil
}

// HandleGetInputItems handles GET /v1/responses/{id}/input_items requests.
func (f *ResponseAPIFilter) HandleGetInputItems(ctx context.Context, responseID string) (*ext_proc.ProcessingResponse, error) {
	if !f.enabled {
		return createResponseAPIError(404, "Response API not enabled"), nil
	}

	// Get response from store
	stored, err := f.store.GetResponse(ctx, responseID)
	if err != nil {
		if errors.Is(err, responsestore.ErrNotFound) {
			logging.Infof("Response API: Response not found: %s", responseID)
			return createResponseAPIError(404, "Response not found: "+responseID), nil
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

	logging.Infof("Response API: Retrieved %d input items for response %s", len(inputItems), responseID)
	return createImmediateJSONResponse(200, body), nil
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
func (f *ResponseAPIFilter) determineConversationID(req *responseapi.ResponseAPIRequest, history []*responseapi.StoredResponse) string {
	// Priority 1: Request explicitly provides conversation_id
	if req.ConversationID != "" {
		return req.ConversationID
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

// getConversationIDSource returns a string describing where the ConversationID came from (for logging).
func (f *ResponseAPIFilter) getConversationIDSource(req *responseapi.ResponseAPIRequest, history []*responseapi.StoredResponse) string {
	if req.ConversationID != "" {
		return "request"
	}
	if len(history) > 0 && history[0].ConversationID != "" {
		return "history"
	}
	return "generated"
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
