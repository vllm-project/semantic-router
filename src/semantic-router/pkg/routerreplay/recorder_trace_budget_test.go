package routerreplay

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func budgetSteps(count, size int) []ToolTraceStep {
	steps := make([]ToolTraceStep, count)
	for i := range steps {
		label := fmt.Sprintf("%02d:", i)
		steps[i] = ToolTraceStep{Type: "user_input", Source: ToolTraceSourceRequest, Text: label + strings.Repeat("x", size-len(label))}
	}
	return steps
}

func stepLabels(trace *ToolTrace) []string {
	labels := make([]string, len(trace.Steps))
	for i, step := range trace.Steps {
		labels[i] = step.Text[:3]
	}
	return labels
}

// The total budget drops the oldest steps first, counts the raw copies,
// always keeps the newest step and adds to the steps the step cap dropped.
func TestCapToolTraceTotalBytesDropsTheOldestSteps(t *testing.T) {
	trace := &ToolTrace{Steps: budgetSteps(10, 1000)}
	capToolTraceTotalBytes(trace, 3500)
	require.Equal(t, []string{"07:", "08:", "09:"}, stepLabels(trace))
	require.True(t, trace.StepsTruncated)
	require.Equal(t, 7, trace.DroppedStepCount)

	calls := &ToolTrace{Steps: []ToolTraceStep{
		{Type: "assistant_tool_call", Arguments: strings.Repeat("a", 1000), RawArguments: strings.Repeat("a", 1000)},
		{Type: "assistant_tool_call", Arguments: strings.Repeat("b", 1000), RawArguments: strings.Repeat("b", 1000)},
		{Type: "assistant_tool_call", Arguments: strings.Repeat("c", 1000), RawArguments: strings.Repeat("c", 1000)},
	}}
	capToolTraceTotalBytes(calls, 4500)
	require.Len(t, calls.Steps, 2, "raw copies count toward the budget")
	require.Equal(t, 1, calls.DroppedStepCount)

	oversized := &ToolTrace{Steps: budgetSteps(3, 10_000)}
	capToolTraceTotalBytes(oversized, 1000)
	require.Equal(t, []string{"02:"}, stepLabels(oversized), "the newest step is kept even alone over the budget")
	require.Equal(t, 2, oversized.DroppedStepCount)

	whole := &ToolTrace{Steps: budgetSteps(10, 1000)}
	capToolTraceTotalBytes(whole, 0)
	require.Len(t, whole.Steps, 10)
	require.False(t, whole.StepsTruncated)

	capped := &ToolTrace{Steps: budgetSteps(10, 1000), StepsTruncated: true, DroppedStepCount: 5}
	capToolTraceTotalBytes(capped, 2000)
	require.Equal(t, 13, capped.DroppedStepCount)
}

// A step's raw copies stay as they came unless a raw limit is set; then they
// are cut on a character boundary and RawTruncated says so.
func TestTruncateToolTraceStepCutsRawCopiesOnlyUnderARawLimit(t *testing.T) {
	arguments := strings.Repeat("参数", 20)
	step := ToolTraceStep{Arguments: arguments, RawArguments: arguments + "\xff", Output: "short", RawOutput: "short"}
	truncateToolTraceStep(&step, 16, 0)
	require.True(t, step.Truncated)
	require.LessOrEqual(t, len(step.Arguments), 16)
	require.Equal(t, arguments+"\xff", step.RawArguments, "raw copies stay whole and as they came without a raw limit")
	require.False(t, step.RawTruncated)

	step = ToolTraceStep{Arguments: arguments, RawArguments: arguments, Output: "short", RawOutput: "short"}
	truncateToolTraceStep(&step, 16, 16)
	require.True(t, step.RawTruncated)
	require.LessOrEqual(t, len(step.RawArguments), 16)
	require.True(t, strings.HasPrefix(arguments, step.RawArguments), "the cut ends on a character boundary")
	require.Equal(t, "short", step.RawOutput)
}

// A record's trace is bounded as its policy says on both write paths: when the
// record is added and when its trace is replaced with the response's.
func TestRecorderBoundsTheTraceOnEveryWritePath(t *testing.T) {
	owner := NewRecorder(store.NewMemoryStore(100, 0))
	t.Cleanup(func() { require.NoError(t, owner.Close()) })
	view := owner.WithCapturePolicy(CapturePolicy{
		CaptureRequestBody: true, CaptureResponseBody: true, MaxBodyBytes: 4096,
		MaxToolTraceBytes: 64, MaxRawToolTraceBytes: 64, MaxToolTraceTotalBytes: 300, MaxToolTraceSteps: 100,
	})
	id, err := view.AddRecord(RoutingRecord{ToolTrace: &ToolTrace{Steps: budgetSteps(10, 100)}})
	require.NoError(t, err)
	record, ok := view.GetRecord(id)
	require.True(t, ok)
	require.Equal(t, []string{"06:", "07:", "08:", "09:"}, stepLabels(record.ToolTrace))
	require.Equal(t, 6, record.ToolTrace.DroppedStepCount)
	for _, step := range record.ToolTrace.Steps {
		require.Len(t, step.Text, 64)
		require.True(t, step.Truncated)
	}

	update := budgetSteps(10, 100)
	update[9] = ToolTraceStep{Type: "assistant_tool_call", Source: ToolTraceSourceResponse, Arguments: strings.Repeat("r", 100), RawArguments: strings.Repeat("r", 100)}
	require.NoError(t, view.UpdateToolTrace(id, ToolTrace{Steps: update}))
	record, ok = view.GetRecord(id)
	require.True(t, ok)
	require.Len(t, record.ToolTrace.Steps, 3, "64 + 64 bytes of the call and two 64-byte turns fit 300 bytes")
	last := record.ToolTrace.Steps[2]
	require.Len(t, last.RawArguments, 64)
	require.True(t, last.RawTruncated)
	require.True(t, record.ToolTrace.StepsTruncated)
}
