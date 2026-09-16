package mcp

import (
	"errors"
	"strings"
	"testing"
)

func TestEnvironmentVariableLogSummaryOmitsValues(t *testing.T) {
	const sentinelValue = "stdio-env-value-canary"

	summary := environmentVariableLogSummary(map[string]string{
		"SECOND_NAME": "ordinary",
		"FIRST_NAME":  sentinelValue,
	})

	if strings.Contains(summary, sentinelValue) || strings.Contains(summary, "FIRST_NAME=") {
		t.Fatalf("environment summary exposed a configured value: %q", summary)
	}
	if summary != "Environment variable names (2): [FIRST_NAME SECOND_NAME]" {
		t.Fatalf("environment summary = %q", summary)
	}
}

func TestToolCallFailureLogMessageOmitsEchoedErrorContent(t *testing.T) {
	const secretCanary = "mcp-error-echo-secret-canary" //nolint:gosec // Canary verifies remote errors are not logged.
	message := toolCallFailureLogMessage(
		"test-tool",
		errors.New("remote server echoed "+secretCanary),
	)

	if strings.Contains(message, secretCanary) {
		t.Fatalf("tool failure log exposed remote error content: %q", message)
	}
	if !strings.Contains(message, "error_class=") {
		t.Fatalf("tool failure log omitted safe error class: %q", message)
	}
}
