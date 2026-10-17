/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package handoff

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var testNow = time.Date(2026, time.September, 4, 12, 0, 0, 0, time.UTC)

// validJSON returns a minimal valid envelope with top-level overrides applied.
// A nil override value deletes the field.
func validJSON(overrides map[string]any) []byte {
	fields := map[string]any{
		"version":            Version,
		"handoff_id":         "handoff-1",
		"root_invocation_id": "inv-root",
		"expires_at":         testNow.Add(5 * time.Minute).Format(time.RFC3339),
	}
	for key, value := range overrides {
		if value == nil {
			delete(fields, key)
			continue
		}
		fields[key] = value
	}
	data, err := json.Marshal(fields)
	if err != nil {
		panic(err)
	}
	return data
}

func TestParseMinimalEnvelopeDefaultsToActive(t *testing.T) {
	envelope, err := Parse(validJSON(nil), testNow)

	require.NoError(t, err)
	assert.Equal(t, Version, envelope.Version)
	assert.Equal(t, "handoff-1", envelope.HandoffID)
	assert.Equal(t, StateActive, envelope.State)
	assert.Nil(t, envelope.Selection)
	assert.Nil(t, envelope.Runtime)
}

func TestParseCompleteEnvelopeSeparatesSelectionFromRuntime(t *testing.T) {
	envelope, err := Parse(validJSON(map[string]any{
		"parent_invocation_id": "inv-parent",
		"state":                "cancelled",
		"selection": map[string]any{
			"delegated_role":        "researcher",
			"required_capabilities": []string{"vision", "tools.calling"},
			"remaining_tokens":      0,
		},
		"runtime": map[string]any{
			"task_summary":    "Summarize the findings.",
			"result_summary":  "Draft ready.",
			"tool_state_refs": []string{"tool-state:one", "tool-state:two"},
		},
	}), testNow)

	require.NoError(t, err)
	assert.Equal(t, StateCancelled, envelope.State)
	require.NotNil(t, envelope.Selection)
	assert.Equal(t, "researcher", envelope.Selection.DelegatedRole)
	assert.Equal(t, []string{"vision", "tools.calling"}, envelope.Selection.RequiredCapabilities)
	require.NotNil(t, envelope.Selection.RemainingTokens, "zero remaining tokens is distinct from absent")
	assert.Equal(t, 0, *envelope.Selection.RemainingTokens)
	require.NotNil(t, envelope.Runtime)
	assert.Equal(t, []string{"tool-state:one", "tool-state:two"}, envelope.Runtime.ToolStateRefs)
}

func TestParseRejectsMalformedAndAmbiguousJSON(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		code ErrorCode
	}{
		{name: "empty", data: nil, code: CodeMalformedJSON},
		{name: "truncated", data: []byte(`{"version":`), code: CodeMalformedJSON},
		{name: "not an object", data: []byte(`["1"]`), code: CodeMalformedJSON},
		{name: "trailing value", data: append(validJSON(nil), []byte(` {}`)...), code: CodeMalformedJSON},
		{name: "duplicate key", data: []byte(`{"version":"1","version":"1"}`), code: CodeMalformedJSON},
		{name: "duplicate nested key", data: validJSON(map[string]any{"selection": json.RawMessage(`{"delegated_role":"a","delegated_role":"b"}`)}), code: CodeMalformedJSON},
		{name: "wrong type", data: validJSON(map[string]any{"handoff_id": 7}), code: CodeMalformedJSON},
		{name: "unknown top-level field", data: validJSON(map[string]any{"future_field": true}), code: CodeUnknownField},
		{name: "unknown nested field", data: validJSON(map[string]any{"runtime": map[string]any{"transcript": "x"}}), code: CodeUnknownField},
		{name: "oversized", data: []byte(`{"version":"1","pad":"` + strings.Repeat("a", MaxJSONBytes) + `"}`), code: CodePayloadTooLarge},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Parse(test.data, testNow)
			require.Error(t, err)
			assert.Equal(t, test.code, CodeOf(err))
		})
	}
}

func TestParseReportsVersionBeforeFieldCompatibility(t *testing.T) {
	newer := validJSON(map[string]any{"version": "2", "future_field": true})

	_, err := Parse(newer, testNow)

	assert.Equal(t, CodeUnsupportedVersion, CodeOf(err), "a newer envelope must degrade to unsupported_version, not unknown_field")

	_, err = Parse(validJSON(map[string]any{"version": nil}), testNow)
	assert.Equal(t, &ContractError{Code: CodeMissingField, Field: "version"}, err)
}

func TestParseRejectsInvalidFields(t *testing.T) {
	longID := strings.Repeat("a", MaxIDLength+1)
	tooMany := make([]string, MaxListItems+1)
	for i := range tooMany {
		tooMany[i] = fmt.Sprintf("cap-%d", i)
	}
	tests := []struct {
		name      string
		overrides map[string]any
		want      *ContractError
	}{
		{"missing handoff id", map[string]any{"handoff_id": nil}, &ContractError{CodeMissingField, "handoff_id"}},
		{"null handoff id", map[string]any{"handoff_id": json.RawMessage(`null`)}, &ContractError{CodeMissingField, "handoff_id"}},
		{"missing root", map[string]any{"root_invocation_id": nil}, &ContractError{CodeMissingField, "root_invocation_id"}},
		{"missing expiry", map[string]any{"expires_at": nil}, &ContractError{CodeMissingField, "expires_at"}},
		{"long id", map[string]any{"handoff_id": longID}, &ContractError{CodeInvalidField, "handoff_id"}},
		{"id charset", map[string]any{"handoff_id": "has space"}, &ContractError{CodeInvalidField, "handoff_id"}},
		{"parent charset", map[string]any{"parent_invocation_id": "-leading"}, &ContractError{CodeInvalidField, "parent_invocation_id"}},
		{"state", map[string]any{"state": "paused"}, &ContractError{CodeInvalidField, "state"}},
		{"role case", map[string]any{"selection": map[string]any{"delegated_role": "Researcher"}}, &ContractError{CodeInvalidField, "selection.delegated_role"}},
		{"duplicate capability", map[string]any{"selection": map[string]any{"required_capabilities": []string{"tools", "tools"}}}, &ContractError{CodeInvalidField, "selection.required_capabilities"}},
		{"too many capabilities", map[string]any{"selection": map[string]any{"required_capabilities": tooMany}}, &ContractError{CodeInvalidField, "selection.required_capabilities"}},
		{"negative tokens", map[string]any{"selection": map[string]any{"remaining_tokens": -1}}, &ContractError{CodeInvalidField, "selection.remaining_tokens"}},
		{"too many tokens", map[string]any{"selection": map[string]any{"remaining_tokens": MaxRemainingTokens + 1}}, &ContractError{CodeInvalidField, "selection.remaining_tokens"}},
		{"long summary", map[string]any{"runtime": map[string]any{"task_summary": strings.Repeat("a", MaxSummaryBytes+1)}}, &ContractError{CodeInvalidField, "runtime.task_summary"}},
		{"tool ref charset", map[string]any{"runtime": map[string]any{"tool_state_refs": []string{"a b"}}}, &ContractError{CodeInvalidField, "runtime.tool_state_refs"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Parse(validJSON(test.overrides), testNow)
			assert.Equal(t, test.want, err)
		})
	}
}

func TestParseEnforcesLifetimeWindow(t *testing.T) {
	tests := []struct {
		name      string
		expiresAt string
		code      ErrorCode
	}{
		{name: "at now is expired", expiresAt: testNow.Format(time.RFC3339), code: CodeExpired},
		{name: "past", expiresAt: testNow.Add(-time.Second).Format(time.RFC3339), code: CodeExpired},
		{name: "beyond max lifetime", expiresAt: testNow.Add(MaxLifetime + time.Second).Format(time.RFC3339), code: CodeInvalidField},
		{name: "not RFC 3339", expiresAt: "2026-09-04 12:05:00", code: CodeInvalidField},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Parse(validJSON(map[string]any{"expires_at": test.expiresAt}), testNow)
			assert.Equal(t, test.code, CodeOf(err))
		})
	}

	envelope, err := Parse(validJSON(map[string]any{
		"expires_at": testNow.Add(MaxLifetime).In(time.FixedZone("UTC+8", 8*3600)).Format(time.RFC3339),
	}), testNow)
	require.NoError(t, err, "the max lifetime boundary is inclusive and offsets are normalized")
	assert.Equal(t, time.UTC, envelope.ExpiresAt.Location())
}

func TestDigestIsStableAcrossEquivalentEncodings(t *testing.T) {
	compact, err := Parse(validJSON(nil), testNow)
	require.NoError(t, err)
	explicit, err := Parse(validJSON(map[string]any{
		"state":      "active",
		"expires_at": testNow.Add(5 * time.Minute).In(time.FixedZone("UTC+8", 8*3600)).Format(time.RFC3339),
	}), testNow)
	require.NoError(t, err)
	changed, err := Parse(validJSON(map[string]any{"root_invocation_id": "inv-other"}), testNow)
	require.NoError(t, err)

	assert.Equal(t, compact.Digest(), explicit.Digest())
	assert.NotEqual(t, compact.Digest(), changed.Digest())
}

func TestContractErrorsAndFormattingNeverExposeValues(t *testing.T) {
	const secret = "secret-summary-value"
	_, err := Parse(validJSON(map[string]any{
		"runtime": map[string]any{"task_summary": secret + strings.Repeat("x", MaxSummaryBytes)},
	}), testNow)
	require.Error(t, err)
	assert.NotContains(t, err.Error(), secret)

	envelope, err := Parse(validJSON(map[string]any{
		"runtime": map[string]any{"task_summary": secret},
	}), testNow)
	require.NoError(t, err)
	assert.NotContains(t, fmt.Sprintf("%v %+v %s", envelope, envelope, envelope), secret)
}
