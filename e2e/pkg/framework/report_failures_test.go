package framework

import (
	"errors"
	"strings"
	"testing"
)

func TestFailedAndSlowestMarkdownListsFailuresFirst(t *testing.T) {
	md := failedAndSlowestMarkdown([]TestResult{
		{Name: "fast", Passed: true, Duration: "1s"},
		{Name: "broken", Passed: false, Duration: "3s", Error: errors.New("raw value reached the backend")},
		{Name: "slow", Passed: true, Duration: "2m0s"},
	})
	failed, slowest := strings.Index(md, "### ❌ Failed"), strings.Index(md, "### ⏱ Slowest")
	if failed < 0 || slowest < failed || !strings.Contains(md, "**broken**: `raw value reached the backend`") {
		t.Fatalf("failures are not listed first:\n%s", md)
	}
	if strings.Index(md, "- slow (2m0s)") > strings.Index(md, "- broken (3s)") {
		t.Fatalf("slowest cases are not ordered by duration:\n%s", md)
	}
	if strings.Contains(failedAndSlowestMarkdown([]TestResult{{Name: "ok", Passed: true}}), "Failed") {
		t.Fatal("a passing run lists failures")
	}
}

func TestFailureAnnotationEscapesWorkflowCommand(t *testing.T) {
	got := failureAnnotation("masking", TestResult{Name: "a:b", Error: errors.New("100% wrong\nsecond line")})
	want := "::error title=E2E masking / a%3Ab::100%25 wrong%0Asecond line"
	if got != want {
		t.Fatalf("annotation = %q, want %q", got, want)
	}
}
