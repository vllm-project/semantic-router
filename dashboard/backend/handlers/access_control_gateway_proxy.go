package handlers

import (
	"bytes"
	"errors"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/accesscontrol"
)

type gatewayStreamResult struct {
	responseBody []byte
	ttftMS       int64
	streamErr    error
}

type gatewayUsageResult struct {
	promptTokens     int64
	completionTokens int64
	totalTokens      int64
	estimated        bool
}

func (h *AccessGatewayHandler) proxyChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	principal *accesscontrol.Principal,
	credentialType string,
	started time.Time,
	requestID string,
	body []byte,
	envelope chatCompletionEnvelope,
	estimatedTokens int64,
	reservation *accesscontrol.Reservation,
) {
	request, err := h.newUpstreamChatRequest(r, principal, requestID, body)
	if err != nil {
		h.failChatProxy(w, principal, credentialType, started, requestID, body, envelope, reservation, "gateway_request_error", "could not create upstream request")
		return
	}
	response, err := h.client.Do(request)
	if err != nil {
		h.failChatProxy(w, principal, credentialType, started, requestID, body, envelope, reservation, "upstream_unavailable", "inference upstream is unavailable")
		return
	}
	defer response.Body.Close()

	stream := writeGatewayResponse(w, response, requestID, envelope.Stream, started)
	usage := resolveGatewayUsage(stream.responseBody, envelope.Stream, response.StatusCode, estimatedTokens)
	h.reconcileQuota(reservation, usage.totalTokens, requestID)
	metadata := gatewayUsageMetadata(body, stream.responseBody, envelope.Stream, credentialType)
	if usage.estimated {
		metadata["usageEstimated"] = true
	}
	h.recordGatewayUsage(
		principal, requestID, envelope.Model, response.StatusCode, started, stream.ttftMS,
		usage.promptTokens, usage.completionTokens, usage.totalTokens,
		gatewayProxyErrorCode(response.StatusCode, stream.streamErr), metadata,
	)
}

func (h *AccessGatewayHandler) newUpstreamChatRequest(
	r *http.Request,
	principal *accesscontrol.Principal,
	requestID string,
	body []byte,
) (*http.Request, error) {
	request, err := http.NewRequestWithContext(r.Context(), http.MethodPost, h.resolve("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", r.Header.Get("Accept"))
	request.Header.Set("X-Request-ID", requestID)
	request.Header.Set("X-VLLM-SR-API-Key-ID", principal.Key.ID)
	if principal.Key.UserID != "" {
		request.Header.Set("X-VLLM-SR-User-ID", principal.Key.UserID)
	}
	if principal.Team != nil && principal.Team.ID != "" {
		request.Header.Set("X-VLLM-SR-Team-ID", principal.Team.ID)
	}
	return request, nil
}

func (h *AccessGatewayHandler) failChatProxy(
	w http.ResponseWriter,
	principal *accesscontrol.Principal,
	credentialType string,
	started time.Time,
	requestID string,
	body []byte,
	envelope chatCompletionEnvelope,
	reservation *accesscontrol.Reservation,
	errorCode string,
	message string,
) {
	h.reconcileQuota(reservation, 0, requestID)
	h.recordGatewayUsage(
		principal, requestID, envelope.Model, http.StatusBadGateway, started, 0, 0, 0, 0,
		errorCode, gatewayUsageMetadata(body, nil, envelope.Stream, credentialType),
	)
	writeAccessError(w, http.StatusBadGateway, message)
}

func writeGatewayResponse(
	w http.ResponseWriter,
	response *http.Response,
	requestID string,
	streaming bool,
	started time.Time,
) gatewayStreamResult {
	copyGatewayResponseHeaders(w, response, requestID, streaming)
	capture := &limitedCapture{limit: maxGatewayBody}
	result := gatewayStreamResult{}
	buffer := make([]byte, 32*1024)
	for {
		read, readErr := response.Body.Read(buffer)
		if read > 0 {
			if result.ttftMS == 0 {
				result.ttftMS = time.Since(started).Milliseconds()
			}
			_, _ = capture.Write(buffer[:read])
			_, _ = w.Write(buffer[:read])
			flushGatewayChunk(w, streaming)
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				result.streamErr = readErr
				log.Printf("access gateway response stream failed for request %s: %v", requestID, readErr)
			}
			break
		}
	}
	result.responseBody = capture.Bytes()
	return result
}

func copyGatewayResponseHeaders(w http.ResponseWriter, response *http.Response, requestID string, streaming bool) {
	for key, values := range response.Header {
		if isGatewayResponseHeader(key) {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}
	w.Header().Set("X-Request-ID", requestID)
	if streaming {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(response.StatusCode)
}

func flushGatewayChunk(w http.ResponseWriter, streaming bool) {
	if flusher, ok := w.(http.Flusher); ok && streaming {
		flusher.Flush()
	}
}

func resolveGatewayUsage(body []byte, streaming bool, statusCode int, estimatedTokens int64) gatewayUsageResult {
	prompt, completion, total := parseGatewayUsage(body, streaming)
	if total == 0 {
		total = prompt + completion
	}
	// A successful stream may omit the optional final usage chunk. Retain the
	// reservation so concurrent replicas cannot weaken the shared TPM limit.
	estimated := total == 0 && statusCode < http.StatusBadRequest
	if estimated {
		total = estimatedTokens
	}
	return gatewayUsageResult{
		promptTokens:     prompt,
		completionTokens: completion,
		totalTokens:      total,
		estimated:        estimated,
	}
}

func gatewayProxyErrorCode(statusCode int, streamErr error) string {
	if streamErr != nil {
		return "upstream_stream_error"
	}
	if statusCode >= http.StatusBadRequest {
		return "upstream_error"
	}
	return ""
}
