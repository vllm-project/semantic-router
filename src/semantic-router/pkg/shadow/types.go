package shadow

// maxResponseBytes bounds reading an upstream body (32 MiB, matching the
// router's per-upstream ceiling) so a huge or malicious response cannot
// exhaust router memory.
const maxResponseBytes int64 = 32 << 20

func truncate(b []byte) string {
	const max = 512
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max])
}

// Outcome is the normalized lifecycle result of one shadow arm attempt
// (issue #3376). Judge and evidence code normalize to these values; the
// multi-arm budget (ShadowBudget) and dispatch live on the shadow_dispatch
// plugin path.
type Outcome string

const (
	OutcomeCompleted Outcome = "completed"
	OutcomeFailed    Outcome = "failed"
	OutcomeTimedOut  Outcome = "timed_out"
	OutcomeCancelled Outcome = "cancelled"
	OutcomeSkipped   Outcome = "skipped"
)

// ArmResult is the normalized outcome of one shadow arm evaluation, shared by
// the judge and evidence layers. Response content is the extracted completion
// text; raw hidden reasoning is intentionally never captured (issue #3376).
type ArmResult struct {
	// Arm identifies the configured arm (stable name, opaque to observers).
	Arm string
	// Model is what the arm actually invoked (kept for operator debugging,
	// excluded from judge input by the blinding layer).
	Model string
	// Outcome is the normalized lifecycle result.
	Outcome Outcome
	// LatencyMS is the observed round-trip on success.
	LatencyMS int64
	// Content is the extracted text of the completion.
	Content string
	// PromptTokens / CompletionTokens are parsed from the arm's usage block.
	PromptTokens     int64
	CompletionTokens int64
	// Err is set for configuration, timeout, cancellation, budget or HTTP
	// failures.
	Err string
}
