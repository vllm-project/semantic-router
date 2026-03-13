package fixtures

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// ResponseAPIRequest represents a Response API request.
type ResponseAPIRequest struct {
	Model              string            `json:"model"`
	Input              interface{}       `json:"input"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	Instructions       string            `json:"instructions,omitempty"`
	Store              *bool             `json:"store,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
}

// ResponseAPIResponse represents a Response API response.
type ResponseAPIResponse struct {
	ID                 string                   `json:"id"`
	Object             string                   `json:"object"`
	CreatedAt          int64                    `json:"created_at"`
	Model              string                   `json:"model"`
	Status             string                   `json:"status"`
	Output             []map[string]interface{} `json:"output"`
	OutputText         string                   `json:"output_text,omitempty"`
	PreviousResponseID string                   `json:"previous_response_id,omitempty"`
	Usage              map[string]interface{}   `json:"usage,omitempty"`
	Instructions       string                   `json:"instructions,omitempty"`
	Metadata           map[string]string        `json:"metadata,omitempty"`
}

// DeleteResponseResult represents the result of deleting a response.
type DeleteResponseResult struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// InputItemsListResponse represents the response for GET /v1/responses/{id}/input_items.
type InputItemsListResponse struct {
	Object  string                   `json:"object"`
	Data    []map[string]interface{} `json:"data"`
	FirstID string                   `json:"first_id"`
	LastID  string                   `json:"last_id"`
	HasMore bool                     `json:"has_more"`
}

// APIErrorResponse represents an error response from the API.
type APIErrorResponse struct {
	Error APIErrorDetail `json:"error"`
}

// APIErrorDetail contains the error details.
type APIErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code,omitempty"`
}

// ResponseAPIClient talks to the OpenAI-compatible Responses API surface.
type ResponseAPIClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewResponseAPIClient binds a Responses client to a port-forward session.
func NewResponseAPIClient(session *ServiceSession, timeout time.Duration) *ResponseAPIClient {
	return &ResponseAPIClient{
		baseURL:    session.BaseURL(),
		httpClient: session.HTTPClient(timeout),
	}
}

// CreateRaw sends a POST /v1/responses request and returns the raw response.
func (c *ResponseAPIClient) CreateRaw(ctx context.Context, request any) (*HTTPResponse, error) {
	return doJSONRequest(ctx, c.httpClient, http.MethodPost, c.baseURL+"/v1/responses", request, nil)
}

// Create sends a typed POST /v1/responses request.
func (c *ResponseAPIClient) Create(
	ctx context.Context,
	request ResponseAPIRequest,
) (*ResponseAPIResponse, *HTTPResponse, error) {
	raw, err := c.CreateRaw(ctx, request)
	if err != nil {
		return nil, nil, err
	}
	var parsed ResponseAPIResponse
	if err := decodeExpectedJSON(raw, http.StatusOK, &parsed); err != nil {
		return nil, raw, err
	}
	return &parsed, raw, nil
}

// GetRaw sends a GET /v1/responses/{id} request.
func (c *ResponseAPIClient) GetRaw(ctx context.Context, responseID string) (*HTTPResponse, error) {
	return doJSONRequest(ctx, c.httpClient, http.MethodGet, c.baseURL+"/v1/responses/"+responseID, nil, nil)
}

// Get retrieves a stored response.
func (c *ResponseAPIClient) Get(
	ctx context.Context,
	responseID string,
) (*ResponseAPIResponse, *HTTPResponse, error) {
	raw, err := c.GetRaw(ctx, responseID)
	if err != nil {
		return nil, nil, err
	}
	var parsed ResponseAPIResponse
	if err := decodeExpectedJSON(raw, http.StatusOK, &parsed); err != nil {
		return nil, raw, err
	}
	return &parsed, raw, nil
}

// DeleteRaw sends a DELETE /v1/responses/{id} request.
func (c *ResponseAPIClient) DeleteRaw(ctx context.Context, responseID string) (*HTTPResponse, error) {
	return doJSONRequest(ctx, c.httpClient, http.MethodDelete, c.baseURL+"/v1/responses/"+responseID, nil, nil)
}

// Delete removes a stored response.
func (c *ResponseAPIClient) Delete(
	ctx context.Context,
	responseID string,
) (*DeleteResponseResult, *HTTPResponse, error) {
	raw, err := c.DeleteRaw(ctx, responseID)
	if err != nil {
		return nil, nil, err
	}
	var parsed DeleteResponseResult
	if err := decodeExpectedJSON(raw, http.StatusOK, &parsed); err != nil {
		return nil, raw, err
	}
	return &parsed, raw, nil
}

// InputItemsRaw sends a GET /v1/responses/{id}/input_items request.
func (c *ResponseAPIClient) InputItemsRaw(ctx context.Context, responseID string) (*HTTPResponse, error) {
	return doJSONRequest(ctx, c.httpClient, http.MethodGet, c.baseURL+"/v1/responses/"+responseID+"/input_items", nil, nil)
}

// InputItems lists stored input items for a response.
func (c *ResponseAPIClient) InputItems(
	ctx context.Context,
	responseID string,
) (*InputItemsListResponse, *HTTPResponse, error) {
	raw, err := c.InputItemsRaw(ctx, responseID)
	if err != nil {
		return nil, nil, err
	}
	var parsed InputItemsListResponse
	if err := decodeExpectedJSON(raw, http.StatusOK, &parsed); err != nil {
		return nil, raw, err
	}
	return &parsed, raw, nil
}

func decodeExpectedJSON(raw *HTTPResponse, expectedStatus int, out any) error {
	if raw.StatusCode != expectedStatus {
		return fmt.Errorf("expected status %d, got %d: %s", expectedStatus, raw.StatusCode, string(raw.Body))
	}
	if err := raw.DecodeJSON(out); err != nil {
		return err
	}
	return nil
}
