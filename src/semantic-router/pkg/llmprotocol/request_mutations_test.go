package llmprotocol

import (
	"testing"
)

func TestValidateBlockableRequestField(t *testing.T) {
	validFields := []string{
		"frequency_penalty",
		"presence_penalty",
		"max_tokens", "max_completion_tokens", "max_output_tokens",
		"n", "candidate_count",
		"response_format", "output_format",
		"seed",
		"stop",
		"temperature",
		"top_p",
		"min_p",
		"repetition_penalty",
		"cache_salt",
		"prompt_cache_key",
		"top_k",
		"tools",
		"tool_choice",
		"parallel_tool_calls",
		"reasoning_effort",
		"reasoning_budget_tokens",
		"reasoning.summary",
		"reasoning.display",
		"metadata",
		"store",
		"auto_store",
		"stream",
		"stream_options",
		"verbosity", "text.verbosity",
		"user",
		"truncation",
		"previous_response_id",
		"conversation",
	}

	for _, f := range validFields {
		if err := ValidateBlockableRequestField(f); err != nil {
			t.Errorf("expected field %q to be valid blockable field, got error: %v", f, err)
		}
	}

	invalidFields := []struct {
		field string
		desc  string
	}{
		{"", "empty field"},
		{"   ", "whitespace field"},
		{"model", "required semantic field model"},
		{"messages", "required semantic field messages"},
		{"logprobs", "unsupported field logprobs"},
		{"top_logprobs", "unsupported field top_logprobs"},
		{"unknown_param", "arbitrary unknown field"},
	}

	for _, tc := range invalidFields {
		if err := ValidateBlockableRequestField(tc.field); err == nil {
			t.Errorf("expected error for %s (%q), got nil", tc.desc, tc.field)
		}
	}
}

func TestBlockRequestFieldForwardedFields(t *testing.T) {
	includeUsage := true
	autoStore := true
	req := &Request{
		TextVerbosity:      "high",
		StreamOptions:      StreamOptions{IncludeUsage: &includeUsage},
		EndUserID:          "user-123",
		Truncation:         "auto",
		PreviousResponseID: "resp-prev",
		ConversationID:     "conv-1",
		AutoStore:          &autoStore,
		ReasoningSummary:   "concise",
		ReasoningDisplay:   "summarized",
	}

	// 1. Block verbosity
	changed, err := BlockRequestField(req, "verbosity")
	if err != nil || !changed || req.TextVerbosity != "" {
		t.Fatalf("failed to block verbosity: changed=%v err=%v req=%+v", changed, err, req)
	}
	changed, err = BlockRequestField(req, "verbosity")
	if err != nil || changed {
		t.Fatalf("expected changed=false on second block: changed=%v err=%v", changed, err)
	}

	// 2. Block text.verbosity
	req.TextVerbosity = "low"
	changed, err = BlockRequestField(req, "text.verbosity")
	if err != nil || !changed || req.TextVerbosity != "" {
		t.Fatalf("failed to block text.verbosity: changed=%v err=%v", changed, err)
	}

	// 3. Block stream_options
	changed, err = BlockRequestField(req, "stream_options")
	if err != nil || !changed || req.StreamOptions.IncludeUsage != nil {
		t.Fatalf("failed to block stream_options: changed=%v err=%v req=%+v", changed, err, req)
	}
	changed, err = BlockRequestField(req, "stream_options")
	if err != nil || changed {
		t.Fatalf("expected changed=false on second stream_options block: changed=%v err=%v", changed, err)
	}

	// 4. Block user
	changed, err = BlockRequestField(req, "user")
	if err != nil || !changed || req.EndUserID != "" {
		t.Fatalf("failed to block user: changed=%v err=%v", changed, err)
	}

	// 5. Block truncation
	changed, err = BlockRequestField(req, "truncation")
	if err != nil || !changed || req.Truncation != "" {
		t.Fatalf("failed to block truncation: changed=%v err=%v", changed, err)
	}

	// 6. Block previous_response_id
	changed, err = BlockRequestField(req, "previous_response_id")
	if err != nil || !changed || req.PreviousResponseID != "" {
		t.Fatalf("failed to block previous_response_id: changed=%v err=%v", changed, err)
	}

	// 7. Block conversation
	changed, err = BlockRequestField(req, "conversation")
	if err != nil || !changed || req.ConversationID != "" {
		t.Fatalf("failed to block conversation: changed=%v err=%v", changed, err)
	}

	// 8. Block auto_store
	changed, err = BlockRequestField(req, "auto_store")
	if err != nil || !changed || req.AutoStore != nil {
		t.Fatalf("failed to block auto_store: changed=%v err=%v", changed, err)
	}

	// 9. Block reasoning.summary
	changed, err = BlockRequestField(req, "reasoning.summary")
	if err != nil || !changed || req.ReasoningSummary != "" {
		t.Fatalf("failed to block reasoning.summary: changed=%v err=%v", changed, err)
	}

	// 10. Block reasoning.display
	changed, err = BlockRequestField(req, "reasoning.display")
	if err != nil || !changed || req.ReasoningDisplay != "" {
		t.Fatalf("failed to block reasoning.display: changed=%v err=%v", changed, err)
	}
}

func TestBlockRequestFieldRejectsUnblockable(t *testing.T) {
	req := &Request{Model: "gpt-4o"}
	if _, err := BlockRequestField(req, "model"); err == nil {
		t.Error("expected error blocking required field model, got nil")
	}
	if _, err := BlockRequestField(req, "messages"); err == nil {
		t.Error("expected error blocking required field messages, got nil")
	}
}
