package router

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func attachRouterReplayResponseRedaction(proxy *httputil.ReverseProxy) {
	if proxy == nil {
		return
	}

	originalModifyResponse := proxy.ModifyResponse
	proxy.ModifyResponse = func(resp *http.Response) error {
		if originalModifyResponse != nil {
			if err := originalModifyResponse(resp); err != nil {
				return err
			}
		}

		return redactRouterReplayResponse(resp)
	}
}

func redactRouterReplayResponse(resp *http.Response) error {
	if !shouldRedactRouterReplayResponse(resp) || requestCanViewReplayFlowDetails(resp.Request) {
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	_ = resp.Body.Close()

	redactedBody, changed, err := redactReplayResponseBody(body)
	if err != nil {
		log.Printf("router replay redaction skipped for %s: %v", resp.Request.URL.Path, err)
		resp.Body = io.NopCloser(bytes.NewReader(body))
		return nil
	}
	if !changed {
		resp.Body = io.NopCloser(bytes.NewReader(body))
		return nil
	}

	resp.Body = io.NopCloser(bytes.NewReader(redactedBody))
	resp.ContentLength = int64(len(redactedBody))
	resp.Header.Set("Content-Length", strconv.Itoa(len(redactedBody)))
	return nil
}

func shouldRedactRouterReplayResponse(resp *http.Response) bool {
	if resp == nil || resp.Request == nil {
		return false
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return false
	}
	if !strings.HasPrefix(resp.Request.URL.Path, "/v1/router_replay") {
		return false
	}
	return strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "application/json")
}

func requestCanViewReplayFlowDetails(r *http.Request) bool {
	if r == nil {
		return false
	}

	ac, ok := auth.AuthFromContext(r)
	if !ok {
		return false
	}
	return ac.Perms[auth.PermConfigWrite]
}

func redactReplayResponseBody(body []byte) ([]byte, bool, error) {
	if len(bytes.TrimSpace(body)) == 0 {
		return body, false, nil
	}

	var payload any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body, false, err
	}

	changed := redactReplayPayload(payload)
	if !changed {
		return body, false, nil
	}

	redactedBody, err := json.Marshal(payload)
	if err != nil {
		return body, false, err
	}
	return redactedBody, true, nil
}

func redactReplayPayload(payload any) bool {
	switch typed := payload.(type) {
	case map[string]any:
		changed := redactReplayRecordMap(typed)

		data, ok := typed["data"].([]any)
		if !ok {
			return changed
		}
		for _, item := range data {
			record, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if redactReplayRecordMap(record) {
				changed = true
			}
		}
		return changed
	case []any:
		changed := false
		for _, item := range typed {
			record, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if redactReplayRecordMap(record) {
				changed = true
			}
		}
		return changed
	default:
		return false
	}
}

func redactReplayRecordMap(record map[string]any) bool {
	changed := false

	if body, ok := record["request_body"].(string); ok && strings.TrimSpace(body) != "" {
		record["request_body"] = ""
		changed = true
	}
	if body, ok := record["response_body"].(string); ok && strings.TrimSpace(body) != "" {
		record["response_body"] = ""
		changed = true
	}

	toolTrace, ok := record["tool_trace"].(map[string]any)
	if !ok {
		return changed
	}
	if redactToolTraceMap(toolTrace) {
		changed = true
	}
	return changed
}

func redactToolTraceMap(toolTrace map[string]any) bool {
	changed := false

	if flow, ok := toolTrace["flow"].(string); ok && strings.TrimSpace(flow) != "" {
		toolTrace["flow"] = ""
		changed = true
	}

	steps, ok := toolTrace["steps"].([]any)
	if !ok {
		return changed
	}

	for _, rawStep := range steps {
		step, ok := rawStep.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := step["text"].(string); ok && strings.TrimSpace(text) != "" {
			step["text"] = ""
			changed = true
		}
		if arguments, ok := step["arguments"].(string); ok && strings.TrimSpace(arguments) != "" {
			step["arguments"] = ""
			changed = true
		}
	}

	return changed
}
