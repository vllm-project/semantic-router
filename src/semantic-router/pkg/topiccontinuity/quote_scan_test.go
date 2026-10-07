package topiccontinuity

import (
	"context"
	"reflect"
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

// A deadline that expires while the live turn's phrases are scanned must
// yield a cancelled result, never an ordinary full-coverage one.
func TestDeadlineDuringEvaluationCancels(t *testing.T) {
	cfg := defaultConfig(adversarialPolicy)
	messages := liveTurnHistory(strings.Repeat(" 'a", MaxTurnBytes/3))
	full := fastestEvaluation(messages, cfg)
	ctx, cancel := context.WithTimeout(context.Background(), full/4)
	defer cancel()
	result := EvaluateAll(ctx, func() ([]llmprotocol.Message, bool) { return messages, true },
		[]EvalConfig{cfg}).Rules[0].Result
	assertInvariants(t, result)
	if result.Reason != ReasonCancelled || result.Coverage == CoverageFull {
		t.Fatalf("deadline %v of a %v evaluation: got %s/%s", full/4, full, result.Reason, result.Coverage)
	}
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

// Cancellation observed at any check, including the final one after the
// phrase scan, yields unknown_cancelled.
func TestCancellationAtEveryCheckPoint(t *testing.T) {
	cfg := defaultConfig(defaultPolicy)
	messages := conversation(unrelatedHistory(3),
		user("New topic: what is the boiling point of water at altitude in 'Denver'?"))
	load := func() ([]llmprotocol.Message, bool) { return messages, true }

	counter := newCountdownContext(1 << 30)
	baseline := EvaluateAll(counter, load, []EvalConfig{cfg}).Rules[0].Result
	checks := (1 << 30) - counter.remaining.Load()
	if baseline.Reason == ReasonCancelled || checks < 2 {
		t.Fatalf("baseline %s after %d checks", baseline.Reason, checks)
	}
	for k := int64(0); k < checks; k++ {
		result := EvaluateAll(newCountdownContext(k), load, []EvalConfig{cfg}).Rules[0].Result
		assertInvariants(t, result)
		if result.Reason != ReasonCancelled {
			t.Fatalf("cancelled at check %d of %d: got %s", k+1, checks, result.Reason)
		}
	}
}
