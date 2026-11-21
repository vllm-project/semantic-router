package extproc

import (
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// traceMembersOf splits an x-vsr-trace value into its members, each key to
// its serialized value with parameters; a bare key is a true boolean, "?1".
func traceMembersOf(t *testing.T, value string) map[string]string {
	t.Helper()
	members := make(map[string]string)
	depth, quoted, start := 0, false, 0
	for i := 0; i <= len(value); i++ {
		if i < len(value) {
			switch c := value[i]; {
			case quoted && c == '\\':
				i++
				continue
			case c == '"':
				quoted = !quoted
				continue
			case quoted:
				continue
			case c == '(':
				depth++
				continue
			case c == ')':
				depth--
				continue
			case c != ',' || depth > 0:
				continue
			}
		}
		member := strings.TrimSpace(value[start:i])
		start = i + 1
		if member == "" {
			continue
		}
		key, rest := member, "?1"
		if at := strings.IndexAny(member, "=;"); at >= 0 {
			key = member[:at]
			if member[at] == '=' {
				rest = member[at+1:]
			} else {
				rest = "?1" + member[at:]
			}
		}
		if _, dup := members[key]; dup {
			t.Fatalf("trace %q repeats member %s", value, key)
		}
		members[key] = rest
	}
	if quoted || depth != 0 {
		t.Fatalf("trace %q is not a well-formed dictionary", value)
	}
	return members
}

// traceOf returns the members of the x-vsr-trace header among set headers,
// nil when none carries it.
func traceOf(t *testing.T, options []*core.HeaderValueOption) map[string]string {
	t.Helper()
	for _, option := range options {
		if option.GetHeader().GetKey() == headers.VSRTrace {
			return traceMembersOf(t, string(option.GetHeader().GetRawValue()))
		}
	}
	return nil
}

func debugRequestContext() *RequestContext {
	return &RequestContext{Headers: map[string]string{headers.VSRDebug: "true"}}
}

func TestResponseTraceIsOnlyForCallersAllowedTheDebugHeaders(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{}, VSRSelectionMethod: "static"}
	if got := responseTrace(ctx); got != "" {
		t.Fatalf("responseTrace() for a caller without debug = %q, want none", got)
	}
	ctx.Headers[headers.VSRDebug] = "true"
	if got := traceMembersOf(t, responseTrace(ctx))[headers.TraceAlgorithm]; got != "static" {
		t.Fatalf("algorithm = %q, want static", got)
	}
}

func TestResponseTraceReportsEveryRecordedFact(t *testing.T) {
	ctx := debugRequestContext()
	recordResponseHeaderFacts(ctx)
	ctx.LooperTrace = newLooperTrace(looperResponseWithLatencyAndUsage())
	ctx.ToolsSelection = &toolsSelection{outcome: toolsSelected, candidates: 4, kept: []string{"search"}, latency: 2 * time.Millisecond}
	ctx.BenchmarkConfigHash, ctx.BenchmarkInferenceCalls = strings.Repeat("a", 64), 1
	trace := traceMembersOf(t, responseTrace(ctx))
	for key, want := range map[string]string{
		headers.TraceConfigHash:       `"` + strings.Repeat("a", 64) + `"`,
		headers.TraceInferenceCalls:   "1",
		headers.TraceConfidence:       "0.91",
		headers.TraceAlgorithm:        "static",
		headers.TraceRoutingLatencyMs: "0.412",
		headers.TraceFallbackAttempts: "2",
		headers.TraceOnUnknown:        `("guarded";policy=no_match)`,
		headers.TraceClientProtocol:   "anthropic",
		headers.TraceUpstreamProtocol: "openai",
		headers.TraceContextTokens:    "1500",
		headers.TraceCategory:         `"math"`,
		headers.TraceReasoning:        "on",
		headers.TraceModality:         "(text image)",
		headers.TraceSystemPrompt:     "?1",
		headers.TraceCacheSimilarity:  "0.97",
		headers.TraceRetention:        "?1;drop;ttl_turns=3;keep_current_model;prefer_prefix=?0",
		headers.TracePromptCache:      "skipped;reason=caller_markers;inserted=1;preserved=1",
		headers.TraceTools:            "selected;candidates=4;kept=1;latency_ms=2.0",
		headers.TraceLearning:         "(protection;action=allow_switch;scope=conversation;reason=switch_allowed)",
		headers.TraceRanking:          `"escalate-hard";reason="higher priority"`,
	} {
		if got := trace[key]; got != want {
			t.Errorf("%s = %s, want %s", key, got, want)
		}
	}
	signals := trace[headers.TraceSignals]
	for _, want := range []string{`"keyword:matched-rule"`, `"projection:matched-rule"`, `"hallucination:matched-rule"`} {
		if !strings.Contains(signals, want) {
			t.Errorf("signals %s lack %s", signals, want)
		}
	}
	if looperTrace := trace[headers.TraceLooper]; !strings.HasPrefix(looperTrace, "(") || !strings.Contains(looperTrace, ";iterations=") {
		t.Errorf("looper = %s, want the models used with their iterations", looperTrace)
	}
}

func TestResponseTraceStaysWithinItsLimit(t *testing.T) {
	ctx := debugRequestContext()
	ctx.VSRSelectionMethod = "static"
	for i := range 400 {
		ctx.VSRMatchedKeywords = append(ctx.VSRMatchedKeywords, "a-long-keyword-rule-name-"+strings.Repeat("x", i%7))
	}
	trace := responseTrace(ctx)
	if len(trace) > headers.TraceLimitBytes {
		t.Fatalf("trace is %d bytes, over %d", len(trace), headers.TraceLimitBytes)
	}
	members := traceMembersOf(t, trace)
	if members[headers.TraceAlgorithm] != "static" || members[headers.TraceTruncated] != "?1" {
		t.Fatalf("trace kept %v, want the algorithm and the truncation marker", members)
	}
	if _, ok := members[headers.TraceSignals]; ok {
		t.Fatal("trace kept the signals that overflow it")
	}
}

func TestResponseTraceNamesTheConfigBoundEvaluationReceipt(t *testing.T) {
	ctx := debugRequestContext()
	ctx.BenchmarkConfigHash = strings.Repeat("b", 64)
	ctx.BenchmarkModelUsage = `{"version":1,"complete":true,"calls":[]}`
	trace := traceMembersOf(t, responseTrace(ctx))
	if got := trace[headers.TraceModelUsage]; got != `"{\"version\":1,\"complete\":true,\"calls\":[]}"` {
		t.Fatalf("model_usage = %s", got)
	}
	if _, ok := trace[headers.TraceInferenceCalls]; ok {
		t.Fatal("a usage receipt also reported a direct inference call")
	}
}

// looperTraceFor is the trace of a Looper answer with the context the
// Looper's own tests use.
func looperTraceFor(t *testing.T, resp *looper.Response, ctx *RequestContext) map[string]string {
	t.Helper()
	ctx.Headers = map[string]string{headers.VSRDebug: "true"}
	ctx.LooperTrace = newLooperTrace(resp)
	return traceMembersOf(t, responseTrace(ctx))
}
