package fixtures

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// HTTPResponse captures the raw HTTP exchange for typed clients.
type HTTPResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

// DecodeJSON unmarshals the response body into the provided value.
func (r *HTTPResponse) DecodeJSON(v any) error {
	if err := json.Unmarshal(r.Body, v); err != nil {
		return fmt.Errorf("decode JSON response: %w", err)
	}
	return nil
}

// DoGETRequest sends a GET request and returns the raw HTTP response.
func DoGETRequest(ctx context.Context, httpClient *http.Client, url string) (*HTTPResponse, error) {
	return doJSONRequest(ctx, httpClient, http.MethodGet, url, nil, nil)
}

// DoPOSTRequest sends a POST request with a JSON payload and returns the raw HTTP response.
func DoPOSTRequest(ctx context.Context, httpClient *http.Client, url string, payload any) (*HTTPResponse, error) {
	return doJSONRequest(ctx, httpClient, http.MethodPost, url, payload, nil)
}

// DoDELETERequest sends a DELETE request and returns the raw HTTP response.
func DoDELETERequest(ctx context.Context, httpClient *http.Client, url string) (*HTTPResponse, error) {
	return doJSONRequest(ctx, httpClient, http.MethodDelete, url, nil, nil)
}

func doJSONRequest(
	ctx context.Context,
	httpClient *http.Client,
	method string,
	url string,
	payload any,
	headers map[string]string,
) (*HTTPResponse, error) {
	var body io.Reader
	if payload != nil {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		body = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	return &HTTPResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header.Clone(),
		Body:       responseBody,
	}, nil
}
