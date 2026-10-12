package topiccontinuity

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// referenceSingleQuoteSpans is the original quadratic scan, kept as the
// behavioral reference for the linear one.
func referenceSingleQuoteSpans(text string) []byteRange {
	var spans []byteRange
	for offset := 0; offset < len(text); {
		value, width := utf8.DecodeRuneInString(text[offset:])
		closing, opens := quoteCloser(value)
		if !opens || !boundaryBefore(text, offset) {
			offset += width
			continue
		}
		end := -1
		for at := offset + width; at < len(text); {
			candidate, size := utf8.DecodeRuneInString(text[at:])
			if candidate == '\n' {
				break
			}
			if candidate == closing && at > offset+width && boundaryAfter(text, at+size) {
				end = at + size
				break
			}
			at += size
		}
		if end < 0 {
			offset += width
			continue
		}
		spans = append(spans, byteRange{Start: offset, End: end})
		offset = end
	}
	return spans
}

var quoteScanCases = []string{
	"",
	"'quoted' text",
	"say 'new topic' please",
	"'a 'b 'c",
	" 'a 'b 'c 'done' 'e",
	"don't 'stop' now",
	"‘curly’ and 'straight'",
	"‘open 'mixed’ close'",
	"'first line\n'second' line",
	"'unclosed\n'closed' here",
	"''",
	"' '",
	"'x''y'",
	"'a' 'b' 'c'",
	"(‘nested ‘quotes’)",
	"'é' 'ü",
	strings.Repeat(" 'a", 50) + " 'z'",
}

func TestSingleQuoteSpansMatchesReference(t *testing.T) {
	for _, text := range quoteScanCases {
		if got, want := singleQuoteSpans(text), referenceSingleQuoteSpans(text); !reflect.DeepEqual(got, want) {
			t.Errorf("%q: spans %v, reference %v", text, got, want)
		}
	}
}

func FuzzSingleQuoteSpansMatchesReference(f *testing.F) {
	for _, text := range quoteScanCases {
		f.Add(text)
	}
	f.Fuzz(func(t *testing.T, text string) {
		if got, want := singleQuoteSpans(text), referenceSingleQuoteSpans(text); !reflect.DeepEqual(got, want) {
			t.Fatalf("%q: spans %v, reference %v", text, got, want)
		}
	})
}

// adversarialPolicy allows the largest live turn the configuration accepts.
var adversarialPolicy = HistoryPolicy{
	Limits:           Limits{MaxPriorTurns: 2, MaxTurnBytes: MaxTurnBytes, MaxInputBytes: 3 * MaxTurnBytes},
	IncludeAssistant: true,
}

func liveTurnHistory(live string) []llmprotocol.Message {
	return conversation(unrelatedHistory(2), user(live))
}

func fastestEvaluation(messages []llmprotocol.Message, cfg EvalConfig) time.Duration {
	load := func() ([]llmprotocol.Message, bool) { return messages, true }
	best := time.Duration(1<<63 - 1)
	for i := 0; i < 3; i++ {
		start := time.Now()
		EvaluateAll(context.Background(), load, []EvalConfig{cfg})
		best = min(best, time.Since(start))
	}
	return best
}

// Quote-shaped live turns at the largest allowed size must cost about as
// much as ordinary text of the same length. The comparison is relative, so
// it holds on slow machines; the quadratic scan was hundreds of times slower.
func TestAdversarialQuotesCostLikeOrdinaryText(t *testing.T) {
	cfg := defaultConfig(adversarialPolicy)
	ordinary := fastestEvaluation(liveTurnHistory(strings.Repeat("abc ", MaxTurnBytes/4)), cfg)
	for _, pattern := range []string{" 'a", " ‘a", " 'new topic", "'a\n"} {
		live := strings.Repeat(pattern, MaxTurnBytes/len(pattern))
		if got := fastestEvaluation(liveTurnHistory(live), cfg); got > 10*ordinary+20*time.Millisecond {
			t.Errorf("%q x %d bytes: %v, ordinary text %v", pattern, len(live), got, ordinary)
		}
	}
}

func assertCancelled(t *testing.T, label string, result Result) {
	t.Helper()
	assertInvariants(t, result)
	if result.Reason != ReasonCancelled || result.Coverage != CoveragePartial || result.Features != (Features{}) {
		t.Fatalf("%s: got %s/%s features=%+v", label, result.Reason, result.Coverage, result.Features)
	}
}

// expiredDeadlineContext models a context whose deadline has passed while its
// timer callback has not yet run, as happens when a CPU-bound evaluation
// holds the only processor: Deadline is in the past, and Err is still nil.
type expiredDeadlineContext struct{ context.Context }

func (expiredDeadlineContext) Deadline() (time.Time, bool) {
	return time.Now().Add(-time.Millisecond), true
}

func TestExpiredDeadlineCancelsWithoutTimerCallback(t *testing.T) {
	cfg := defaultConfig(adversarialPolicy)
	messages := liveTurnHistory(strings.Repeat(" 'a", MaxTurnBytes/3))
	ctx := expiredDeadlineContext{context.Background()}
	if ctx.Err() != nil {
		t.Fatal("precondition: Err must still be nil")
	}
	result := EvaluateAll(ctx, func() ([]llmprotocol.Message, bool) { return messages, true },
		[]EvalConfig{cfg}).Rules[0].Result
	assertCancelled(t, "EvaluateAll", result)

	prepared := prepare(context.Background(), messages, true, cfg.Policy)
	assertCancelled(t, "extract", classify(cfg, prepared, extract(ctx, prepared)))
}

// The real-deadline case the maintainer reported, on one CPU: a deadline that
// passes during evaluation must cancel, even before its timer callback runs.
func TestDeadlineDuringEvaluationCancels(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	cfg := defaultConfig(adversarialPolicy)
	messages := liveTurnHistory(strings.Repeat(" 'a", MaxTurnBytes/3))
	full := fastestEvaluation(messages, cfg)
	ctx, cancel := context.WithTimeout(context.Background(), full/4)
	defer cancel()
	result := EvaluateAll(ctx, func() ([]llmprotocol.Message, bool) { return messages, true },
		[]EvalConfig{cfg}).Rules[0].Result
	assertCancelled(t, fmt.Sprintf("deadline %v of a %v evaluation", full/4, full), result)
}

// countdownContext reports cancellation from its k-th Err call onward.
type countdownContext struct {
	context.Context
	remaining atomic.Int64
}

func newCountdownContext(k int64) *countdownContext {
	ctx := &countdownContext{Context: context.Background()}
	ctx.remaining.Store(k)
	return ctx
}

func (c *countdownContext) Err() error {
	if c.remaining.Add(-1) < 0 {
		return context.Canceled
	}
	return nil
}

// Cancellation observed at any probe, including every pass boundary of the
// phrase stage, yields unknown_cancelled with partial coverage and no
// features. The live turn has two segments so per-segment probes run twice.
func TestCancellationAtEveryCheckPoint(t *testing.T) {
	cfg := defaultConfig(defaultPolicy)
	live := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
		text("New topic: what is the boiling point of water at altitude in 'Denver'?"),
		text("Also compare it with \"sea level\" values."),
	}}
	messages := append(unrelatedHistory(3), live)
	load := func() ([]llmprotocol.Message, bool) { return messages, true }

	counter := newCountdownContext(1 << 30)
	baseline := EvaluateAll(counter, load, []EvalConfig{cfg}).Rules[0].Result
	checks := (1 << 30) - counter.remaining.Load()
	if baseline.Reason == ReasonCancelled || checks < 2 {
		t.Fatalf("baseline %s after %d checks", baseline.Reason, checks)
	}
	for k := int64(0); k < checks; k++ {
		result := EvaluateAll(newCountdownContext(k), load, []EvalConfig{cfg}).Rules[0].Result
		assertCancelled(t, fmt.Sprintf("cancelled at check %d of %d", k+1, checks), result)
	}
}
