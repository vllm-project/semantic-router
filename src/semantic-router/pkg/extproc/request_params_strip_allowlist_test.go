//go:build !windows && cgo

package extproc

import "testing"

// strip_unknown must remove genuinely unknown fields but must NOT drop valid
// OpenAI request parameters or the reasoning parameter (chat_template_kwargs)
// that the router itself injects upstream.
func TestStripUnknownKeepsValidParams(t *testing.T) {
	keep := []string{
		"max_completion_tokens", // current OpenAI standard (replaces max_tokens)
		"stream_options",        // OpenAI streaming usage
		"parallel_tool_calls",   // OpenAI tools
		"chat_template_kwargs",  // injected by the router's reasoning mutation
	}
	body := map[string]interface{}{
		"model":               "m",
		"messages":            []interface{}{},
		"bogus_unknown_field": 1,
	}
	for _, k := range keep {
		body[k] = true
	}

	stripUnknownFields(body, true, "d1")

	for _, k := range keep {
		if _, ok := body[k]; !ok {
			t.Errorf("strip_unknown wrongly removed valid field %q", k)
		}
	}
	if _, ok := body["bogus_unknown_field"]; ok {
		t.Error("strip_unknown should still remove genuinely unknown fields")
	}
}
