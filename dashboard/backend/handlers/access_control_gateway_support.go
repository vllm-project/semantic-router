package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
)

const maxGatewayLogPayload = 128 << 10

func gatewayUsageMetadata(requestBody, responseBody []byte, stream bool, credentialType string) map[string]any {
	metadata := map[string]any{"endpoint": "/v1/chat/completions", "stream": stream, "credentialType": credentialType}
	if len(requestBody) > 0 {
		value, truncated := gatewayLogPayload(requestBody, false)
		metadata["request"] = value
		if truncated {
			metadata["requestTruncated"] = true
		}
	}
	if len(responseBody) > 0 {
		value, truncated := gatewayLogPayload(responseBody, stream)
		metadata["response"] = value
		if truncated {
			metadata["responseTruncated"] = true
		}
	}
	return metadata
}

func requestBodyWithStreamUsage(body []byte) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	options, _ := payload["stream_options"].(map[string]any)
	if options == nil {
		options = map[string]any{}
	}
	options["include_usage"] = true
	payload["stream_options"] = options
	return json.Marshal(payload)
}

func gatewayLogPayload(body []byte, preserveText bool) (any, bool) {
	truncated := len(body) > maxGatewayLogPayload
	if truncated {
		body = body[:maxGatewayLogPayload]
	}
	if !preserveText {
		var value any
		if json.Unmarshal(body, &value) == nil {
			return value, truncated
		}
	}
	return string(body), truncated
}

func (h *AccessGatewayHandler) resolve(path string) string {
	target := *h.upstream
	target.Path = strings.TrimRight(target.Path, "/") + path
	target.RawQuery = ""
	return target.String()
}

func parseGatewayUsage(body []byte, stream bool) (int64, int64, int64) {
	type usageEnvelope struct {
		Usage struct {
			Prompt     int64 `json:"prompt_tokens"`
			Completion int64 `json:"completion_tokens"`
			Total      int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	parse := func(raw []byte) (int64, int64, int64) {
		var value usageEnvelope
		if json.Unmarshal(raw, &value) != nil {
			return 0, 0, 0
		}
		return value.Usage.Prompt, value.Usage.Completion, value.Usage.Total
	}
	if !stream {
		return parse(body)
	}
	var prompt, completion, total int64
	for _, line := range bytes.Split(body, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		raw := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if bytes.Equal(raw, []byte("[DONE]")) {
			continue
		}
		p, c, t := parse(raw)
		if t > 0 || p > 0 || c > 0 {
			prompt, completion, total = p, c, t
		}
	}
	return prompt, completion, total
}

func copyGatewayResponse(w http.ResponseWriter, response *http.Response, body []byte) {
	for key, values := range response.Header {
		if isGatewayResponseHeader(key) {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = w.Write(body)
}

func isGatewayResponseHeader(key string) bool {
	normalized := strings.ToLower(key)
	if strings.HasPrefix(normalized, "x-vsr-") {
		return true
	}
	switch normalized {
	case "content-type", "cache-control", "x-request-id", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset":
		return true
	default:
		return false
	}
}

type limitedCapture struct {
	buffer bytes.Buffer
	limit  int64
}

func (w *limitedCapture) Write(value []byte) (int, error) {
	remaining := w.limit - int64(w.buffer.Len())
	if remaining > 0 {
		_, _ = w.buffer.Write(value[:min(int64(len(value)), remaining)])
	}
	return len(value), nil
}

func (w *limitedCapture) Bytes() []byte { return w.buffer.Bytes() }
