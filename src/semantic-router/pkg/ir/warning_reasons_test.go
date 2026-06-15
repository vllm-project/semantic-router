package ir

import "testing"

func TestWarningSeverity_String(t *testing.T) {
	cases := []struct {
		severity WarningSeverity
		want     string
	}{
		{WarningSeverityInfo, "info"},
		{WarningSeverityLossy, "lossy"},
		{WarningSeverityError, "error"},
		{WarningSeverity(99), "unknown"},
	}
	for _, tc := range cases {
		if got := tc.severity.String(); got != tc.want {
			t.Errorf("WarningSeverity(%d).String() = %q, want %q", tc.severity, got, tc.want)
		}
	}
}

func TestWarningReason_StableTokens(t *testing.T) {
	// Locks the initial vocabulary in PR3. Renaming an existing reason is
	// a breaking change for clients pattern-matching on header values.
	expected := map[WarningReason]string{
		ReasonDropped:                       "dropped",
		ReasonCoercedString:                 "coerced_string",
		ReasonUnsupportedBlockType:          "unsupported_block_type",
		ReasonUnsupportedSubtype:            "unsupported_subtype",
		ReasonImageFileIDUnresolved:         "image_file_id_unresolved",
		ReasonRedactedThinkingDropped:       "redacted_thinking_dropped",
		ReasonSignatureDropOnOpenAIBackend:  "signature_drop_on_openai_backend",
		ReasonServerToolDropOnOpenAIBackend: "server_tool_drop_on_openai_backend",
		ReasonTopKDropOnOpenAIBackend:       "top_k_drop_on_openai_backend",
		ReasonBetaDropOnOpenAIBackend:       "beta_drop_on_openai_backend",
		ReasonWarningsTruncated:             "warnings_truncated",
		ReasonCacheFieldsAbsent:             "cache_fields_absent",
		ReasonAnthropicStopReasonCoerced:    "anthropic_stop_reason_coerced",
	}
	for reason, want := range expected {
		if string(reason) != want {
			t.Errorf("WarningReason value drift: got %q, want %q", string(reason), want)
		}
	}
}
