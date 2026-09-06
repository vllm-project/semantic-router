package router

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
)

const (
	redactedToolResultStatusSucceeded = redaction.RedactedToolResultStatusSucceeded
	redactedToolResultStatusFailed    = redaction.RedactedToolResultStatusFailed
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
		return fmt.Errorf("redact Router Replay response for %s: %w", resp.Request.URL.Path, err)
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
	path := resp.Request.URL.Path
	return path == "/v1/router_replay" || strings.HasPrefix(path, "/v1/router_replay/")
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
	return redaction.RedactResponseBody(body)
}

func toolTraceResultStatus(rawText any) string {
	return redaction.ToolTraceResultStatus(rawText)
}
