package router

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const (
	redactedToolResultStatusSucceeded = "succeeded"
	redactedToolResultStatusFailed    = "failed"
	toolExecutionFailurePrefix        = "Tool execution failed:"
	// The router's ext-proc API rejects immediate Replay responses above 4 MiB.
	// Mirror that contract before buffering a response for low-privilege redaction.
	routerReplayRedactionResponseByteLimit int64 = 4 << 20
)

var (
	errRouterReplayResponseTooLarge = errors.New("router replay response exceeds the size limit")
	errRouterReplayRedactionFailed  = errors.New("router replay response redaction failed")
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

	body, err := readBoundedRouterReplayResponseBody(resp.Body)
	if err != nil {
		return err
	}

	redactedBody, changed, err := redactReplayResponseBody(body)
	if err != nil {
		// A low-privilege response must never fail open to the unredacted body.
		return errRouterReplayRedactionFailed
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

func readBoundedRouterReplayResponseBody(body io.ReadCloser) ([]byte, error) {
	defer body.Close()

	limited := &io.LimitedReader{
		R: body,
		N: routerReplayRedactionResponseByteLimit + 1,
	}
	contents, err := io.ReadAll(limited)
	if err != nil {
		return nil, errRouterReplayRedactionFailed
	}
	if int64(len(contents)) > routerReplayRedactionResponseByteLimit {
		return nil, errRouterReplayResponseTooLarge
	}
	return contents, nil
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

	steps, ok := toolTrace["steps"].([]any)
	if !ok {
		return changed
	}

	for _, rawStep := range steps {
		step, ok := rawStep.(map[string]any)
		if !ok {
			continue
		}
		if toolTraceStepType(step) == "client_tool_result" {
			step["status"] = toolTraceResultStatus(step["text"])
			changed = true
		}
		if text, ok := step["text"].(string); ok && strings.TrimSpace(text) != "" {
			step["text"] = ""
			step["content_redacted"] = true
			changed = true
		}
		if arguments, ok := step["arguments"].(string); ok && strings.TrimSpace(arguments) != "" {
			step["arguments"] = ""
			step["content_redacted"] = true
			changed = true
		}
		if rawArguments, ok := step["raw_arguments"].(string); ok && strings.TrimSpace(rawArguments) != "" {
			step["raw_arguments"] = ""
			step["content_redacted"] = true
			changed = true
		}
		if rawOutput, ok := step["raw_output"].(string); ok && strings.TrimSpace(rawOutput) != "" {
			step["raw_output"] = ""
			step["content_redacted"] = true
			changed = true
		}
	}

	return changed
}

func toolTraceStepType(step map[string]any) string {
	value, _ := step["type"].(string)
	return strings.TrimSpace(value)
}

func toolTraceResultStatus(rawText any) string {
	text, _ := rawText.(string)
	normalized := strings.TrimSpace(text)
	if normalized == "" {
		return redactedToolResultStatusFailed
	}

	lower := strings.ToLower(normalized)
	if lower == "null" || lower == "undefined" {
		return redactedToolResultStatusFailed
	}
	if strings.HasPrefix(lower, strings.ToLower(toolExecutionFailurePrefix)) {
		return redactedToolResultStatusFailed
	}
	return redactedToolResultStatusSucceeded
}
