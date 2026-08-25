package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type protocolMatrixHTTPResult struct {
	statusCode  int
	contentType string
	body        []byte
}

func requestProtocolMatrix(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	format string,
	model string,
	stream bool,
) (protocolMatrixHTTPResult, error) {
	path := ""
	var payload map[string]interface{}
	switch format {
	case matrixOpenAIChat:
		path = "/v1/chat/completions"
		payload = map[string]interface{}{
			"model": model, "stream": stream, "max_tokens": 64,
			"messages": []map[string]string{{"role": "user", "content": "Identify the upstream protocol."}},
		}
	case matrixOpenAIResponses:
		path = "/v1/responses"
		payload = map[string]interface{}{
			"model": model, "stream": stream,
			"max_output_tokens": 64, "input": "Identify the upstream protocol.",
		}
	case matrixAnthropic:
		path = "/v1/messages"
		payload = map[string]interface{}{
			"model": model, "stream": stream, "max_tokens": 64,
			"messages": []map[string]string{{"role": "user", "content": "Identify the upstream protocol."}},
		}
	default:
		return protocolMatrixHTTPResult{}, fmt.Errorf("unsupported client format %q", format)
	}
	return executeProtocolMatrixRequest(ctx, client, baseURL+path, format, stream, payload, "")
}

func requestProtocolToolMatrix(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	format string,
	model string,
	stream bool,
) (protocolMatrixHTTPResult, error) {
	path := ""
	parameters := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{"protocol": map[string]string{"type": "string"}},
		"required":   []string{"protocol"},
	}
	var payload map[string]interface{}
	switch format {
	case matrixOpenAIChat:
		path = "/v1/chat/completions"
		payload = map[string]interface{}{
			"model": model, "stream": stream, "max_tokens": 64,
			"messages":    []map[string]string{{"role": "user", "content": "Call protocol_marker."}},
			"tools":       []map[string]interface{}{{"type": "function", "function": map[string]interface{}{"name": "protocol_marker", "description": "Report backend protocol", "parameters": parameters}}},
			"tool_choice": map[string]interface{}{"type": "function", "function": map[string]string{"name": "protocol_marker"}},
		}
	case matrixOpenAIResponses:
		path = "/v1/responses"
		payload = map[string]interface{}{
			"model": model, "stream": stream, "max_output_tokens": 64, "input": "Call protocol_marker.",
			"tools":       []map[string]interface{}{{"type": "function", "name": "protocol_marker", "description": "Report backend protocol", "parameters": parameters}},
			"tool_choice": map[string]string{"type": "function", "name": "protocol_marker"},
		}
	case matrixAnthropic:
		path = "/v1/messages"
		payload = map[string]interface{}{
			"model": model, "stream": stream, "max_tokens": 64,
			"messages":    []map[string]string{{"role": "user", "content": "Call protocol_marker."}},
			"tools":       []map[string]interface{}{{"name": "protocol_marker", "description": "Report backend protocol", "input_schema": parameters}},
			"tool_choice": map[string]string{"type": "tool", "name": "protocol_marker"},
		}
	default:
		return protocolMatrixHTTPResult{}, fmt.Errorf("unsupported client format %q", format)
	}
	return executeProtocolMatrixRequest(ctx, client, baseURL+path, format, stream, payload, "tool ")
}

func executeProtocolMatrixRequest(
	ctx context.Context,
	client *http.Client,
	endpoint string,
	format string,
	stream bool,
	payload map[string]interface{},
	errorPrefix string,
) (protocolMatrixHTTPResult, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return protocolMatrixHTTPResult{}, fmt.Errorf("marshal %srequest: %w", errorPrefix, err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return protocolMatrixHTTPResult{}, fmt.Errorf("create %srequest: %w", errorPrefix, err)
	}
	request.Header.Set("Content-Type", "application/json")
	if stream {
		request.Header.Set("Accept", "text/event-stream")
	}
	if format == matrixAnthropic {
		request.Header.Set("Anthropic-Version", "2023-06-01")
	}
	response, err := client.Do(request)
	if err != nil {
		return protocolMatrixHTTPResult{}, fmt.Errorf("send %srequest: %w", errorPrefix, err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return protocolMatrixHTTPResult{}, fmt.Errorf("read %sresponse: %w", errorPrefix, err)
	}
	return protocolMatrixHTTPResult{
		statusCode:  response.StatusCode,
		contentType: response.Header.Get("Content-Type"),
		body:        responseBody,
	}, nil
}
