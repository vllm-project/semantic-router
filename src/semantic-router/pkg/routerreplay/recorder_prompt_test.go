package routerreplay

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func promptRecord(t *testing.T, recorder *Recorder, prompt string) RoutingRecord {
	t.Helper()
	id, err := recorder.AddRecord(RoutingRecord{RequestID: "prompt-limit", Prompt: prompt, ToolTrace: &ToolTrace{
		Steps: []ToolTraceStep{{Type: "user_input", Text: prompt}},
	}})
	require.NoError(t, err)
	record, found := recorder.GetRecord(id)
	require.True(t, found)
	return record
}

func sha256Hex(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// A prompt longer than the limit is cut to it, and the record says so: the
// whole prompt's size and SHA-256 identify what was cut.
func TestRecorderCutsThePromptAtMaxPromptBytes(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxPromptBytes(10)
	prompt := "Summarize the meeting agenda."
	record := promptRecord(t, recorder, prompt)
	require.Equal(t, prompt[:10], record.Prompt)
	require.True(t, record.PromptTruncated)
	require.Equal(t, len(prompt), record.PromptOriginalBytes)
	require.Equal(t, sha256Hex(prompt), record.PromptOriginalSHA256)
}

// The cut ends on a character boundary, so the prompt kept is valid UTF-8
// and no longer than the limit.
func TestRecorderCutsThePromptOnACharacterBoundary(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxPromptBytes(7)
	prompt := "日本語のテキスト" // three bytes a character
	record := promptRecord(t, recorder, prompt)
	require.Equal(t, "日本", record.Prompt)
	require.True(t, utf8.ValidString(record.Prompt))
	require.True(t, record.PromptTruncated)
	require.Equal(t, len(prompt), record.PromptOriginalBytes)
	require.Equal(t, sha256Hex(prompt), record.PromptOriginalSHA256)
}

// A prompt within the limit is kept whole, with no flags.
func TestRecorderKeepsAPromptWithinTheLimitWhole(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxPromptBytes(64)
	record := promptRecord(t, recorder, "Summarize the meeting agenda.")
	require.Equal(t, "Summarize the meeting agenda.", record.Prompt)
	require.False(t, record.PromptTruncated)
	require.Zero(t, record.PromptOriginalBytes)
	require.Empty(t, record.PromptOriginalSHA256)
}

// A recorder keeps 64 KiB of a prompt by default, and max_prompt_bytes: 0
// keeps the whole prompt.
func TestRecorderPromptLimitDefaultsTo64KiBAndZeroKeepsTheWholePrompt(t *testing.T) {
	prompt := strings.Repeat("All work and no play. ", 5_000)
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	record := promptRecord(t, recorder, prompt)
	require.Len(t, record.Prompt, DefaultMaxPromptBytes)
	require.True(t, record.PromptTruncated)
	require.Equal(t, len(prompt), record.PromptOriginalBytes)

	recorder.SetMaxPromptBytes(0)
	record = promptRecord(t, recorder, prompt)
	require.Equal(t, prompt, record.Prompt)
	require.False(t, record.PromptTruncated)
	require.Zero(t, record.PromptOriginalBytes)
}

// The prompt limit leaves tool traces to max_tool_trace_bytes: a user turn
// in the trace is bounded by that limit alone.
func TestRecorderPromptLimitLeavesTracesToMaxToolTraceBytes(t *testing.T) {
	prompt := "Summarize the meeting agenda."
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxPromptBytes(5)
	record := promptRecord(t, recorder, prompt)
	require.Equal(t, prompt[:5], record.Prompt)
	require.Equal(t, prompt, record.ToolTrace.Steps[0].Text)
	require.False(t, record.ToolTrace.Steps[0].Truncated)

	recorder.SetMaxToolTraceBytes(9)
	record = promptRecord(t, recorder, prompt)
	require.Equal(t, prompt[:5], record.Prompt)
	require.Equal(t, prompt[:9], record.ToolTrace.Steps[0].Text)
}

// A prompt with malformed UTF-8 is normalized before it is cut, and its size
// and hash are those of the normalized prompt, as it would be stored whole.
func TestRecorderHashesThePromptAsItWouldBeStored(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxPromptBytes(4)
	record := promptRecord(t, recorder, "abc\xffdefgh")
	normalized := "abc\uFFFDdefgh"
	require.Equal(t, "abc", record.Prompt)
	require.Equal(t, len(normalized), record.PromptOriginalBytes)
	require.Equal(t, sha256Hex(normalized), record.PromptOriginalSHA256)
}
