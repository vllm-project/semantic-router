package handlers

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

var workerChatEndpointCandidates = []string{
	"/v1/chat/completions",
	"/api/openai/v1/chat/completions",
}

const (
	openClawWorkerChatRequestTimeout         = 10 * time.Minute
	openClawWorkerEndpointRecoveryTimeout    = 10 * time.Minute
	openClawWorkerEndpointRecoveryPollPeriod = 500 * time.Millisecond
)

func newOpenClawWorkerChatHTTPClient() *http.Client {
	return &http.Client{Timeout: openClawWorkerChatRequestTimeout}
}

type workerChatAttemptFailure struct {
	endpoint   string
	statusCode int
	detail     string
}

func buildWorkerChatAttemptFailure(
	endpoint string,
	statusCode int,
	body string,
	err error,
) workerChatAttemptFailure {
	detail := strings.TrimSpace(body)
	if detail == "" && err != nil {
		detail = strings.TrimSpace(err.Error())
	}
	if detail == "" {
		detail = http.StatusText(statusCode)
	}
	if detail == "" {
		detail = "request failed"
	}
	return workerChatAttemptFailure{
		endpoint:   endpoint,
		statusCode: statusCode,
		detail:     detail,
	}
}

func workerChatAllEndpointsMissing(failures []workerChatAttemptFailure) bool {
	if len(failures) == 0 {
		return false
	}
	for _, failure := range failures {
		if failure.statusCode != http.StatusNotFound && failure.statusCode != http.StatusMethodNotAllowed {
			return false
		}
	}
	return true
}

func formatWorkerChatAttemptError(prefix string, failures []workerChatAttemptFailure) error {
	if len(failures) == 0 {
		return fmt.Errorf("%s request failed for all candidate endpoints", prefix)
	}

	parts := make([]string, 0, len(failures))
	for _, failure := range failures {
		part := failure.endpoint + ": " + failure.detail
		if failure.statusCode > 0 {
			part = fmt.Sprintf("%s (%d)", part, failure.statusCode)
		}
		parts = append(parts, part)
	}
	return fmt.Errorf("%s failed across candidate endpoints: %s", prefix, strings.Join(parts, "; "))
}
