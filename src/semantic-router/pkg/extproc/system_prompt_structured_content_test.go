//go:build !windows && cgo

package extproc

import (
	"strings"
	"testing"
)

// insert mode must preserve the existing system message's content even when it
// is structured (an array of content parts), which is valid OpenAI input.
// Previously the content was coerced to "" and silently dropped.
func TestSystemPromptInsertPreservesStructuredContent(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"ORIG_SYS_STRUCTURED"}]},{"role":"user","content":"hi"}]}`)
	out, injected, err := addSystemPromptToRequestBody(body, "INJECTED_PROMPT", "insert")
	if err != nil {
		t.Fatalf("addSystemPromptToRequestBody: %v", err)
	}
	if !injected {
		t.Fatal("expected injection to occur")
	}
	s := string(out)
	if !strings.Contains(s, "INJECTED_PROMPT") {
		t.Fatalf("injected prompt missing: %s", s)
	}
	if !strings.Contains(s, "ORIG_SYS_STRUCTURED") {
		t.Fatalf("original structured system content was silently dropped: %s", s)
	}
}

// Regression: string system content must keep working (prepend + preserve).
func TestSystemPromptInsertPreservesStringContent(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"system","content":"ORIG_STR"},{"role":"user","content":"hi"}]}`)
	out, _, err := addSystemPromptToRequestBody(body, "INJECTED_PROMPT", "insert")
	if err != nil {
		t.Fatalf("addSystemPromptToRequestBody: %v", err)
	}
	s := string(out)
	if !strings.Contains(s, "INJECTED_PROMPT") || !strings.Contains(s, "ORIG_STR") {
		t.Fatalf("string-content insert should keep both prompt and original: %s", s)
	}
}
