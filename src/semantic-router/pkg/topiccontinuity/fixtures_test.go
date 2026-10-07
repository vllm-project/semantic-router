package topiccontinuity

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type goldenCase struct {
	name       string
	messages   []llmprotocol.Message
	policy     *HistoryPolicy
	class      Class
	reason     Reason
	confidence float64 // checked only when hasConf
	hasConf    bool
	coverage   Coverage // checked only when non-empty
}

func smallPolicy() *HistoryPolicy {
	policy := HistoryPolicy{
		Limits:           Limits{MaxPriorTurns: 8, MaxTurnBytes: 256, MaxInputBytes: 1024},
		IncludeAssistant: true,
	}
	return &policy
}

func goldenCases() []goldenCase {
	refactor := exchange("Refactor the routing module so plugins load lazily",
		"Done. The loader now defers plugin initialization until first use.")
	auth := exchange("There is a crash in auth.ts inside validateToken when the header is empty",
		"The crash comes from validateToken reading a missing header.")
	filler := strings.Repeat("lorem ipsum dolor sit amet ", 40)
	return []goldenCase{
		{
			name: "paraphrased continuation",
			messages: conversation(
				exchange("The hot loop in the parser allocates too much memory per request",
					"You could reuse buffers in the hot loop of the parser."),
				user("make the hot loop allocate less memory")),
			class: ClassContinuation, reason: ReasonLexicalOverlap,
			confidence: 0.5 + 0.5*(0.5-0.35)/0.65, hasConf: true, coverage: CoverageFull,
		},
		{
			name:     "entity carry-over",
			messages: conversation(auth, user("Can you add a unit test for validateToken in auth.ts")),
			class:    ClassContinuation, reason: ReasonEntityOverlap, coverage: CoverageFull,
		},
		{
			name: "tool exchange",
			messages: conversation(refactor, user("List the plugin files"),
				assistantCalls(toolCall("t1", "list_files", `{"dir":"plugins"}`)), toolResult("t1", "a.go b.go")),
			class: ClassContinuation, reason: ReasonToolExchange, confidence: 1, hasConf: true,
		},
		{
			name: "orphan tool result",
			messages: conversation(refactor, user("List the plugin files"),
				assistantCalls(toolCall("t1", "list_files", "")), toolResult("t9", "a.go")),
			class: ClassUnknown, reason: ReasonOrphanToolResult, coverage: CoveragePartial,
		},
		{
			name:     "tool result without a user turn",
			messages: conversation(assistantCalls(toolCall("t1", "list_files", "")), toolResult("t1", "a.go")),
			class:    ClassUnknown, reason: ReasonOrphanToolResult,
		},
		{
			name:     "lexically distant follow-up",
			messages: conversation(refactor, user("Good. Now just add a short comment above the loader.")),
			class:    ClassContinuation, reason: ReasonReference, confidence: 0.8, hasConf: true,
		},
		{
			name: "ambiguous follow-up",
			messages: conversation(exchange("Fix the null-pointer crash in auth.ts", "Fixed the nil check."),
				user("Done. Also fix the typo in the README title.")),
			class: ClassUnknown, reason: ReasonInconclusive,
		},
		{
			name:     "abrupt change with full coverage",
			messages: conversation(refactor, user("What is the capital city of Peru and its current population?")),
			class:    ClassChange, reason: ReasonDisjoint, confidence: 0.6, hasConf: true, coverage: CoverageFull,
		},
		{
			name: "abrupt change beyond window",
			messages: conversation(unrelatedHistory(10),
				user("What is the capital city of Peru and its current population?")),
			class: ClassUnknown, reason: ReasonHistoryBeyondWindow, coverage: CoverageWindow,
		},
		{
			name: "old overlap inside window never becomes change",
			messages: conversation(auth, unrelatedHistory(6),
				user("Back in auth.ts, rename validateToken to checkToken")),
			class: ClassUnknown, reason: ReasonInconclusive,
		},
		{
			name: "hidden overlap by truncation",
			messages: conversation(
				exchange(filler+" the helper fooBarBaz lives here "+filler, "ok"),
				user("Refactor fooBarBaz into smaller helper functions today")),
			policy: smallPolicy(), class: ClassUnknown, reason: ReasonIncompleteHistory, coverage: CoveragePartial,
		},
		{
			name:     "explicit marker",
			messages: conversation(refactor, user("Unrelated question: how do I renew a passport?")),
			class:    ClassChange, reason: ReasonExplicitChange, confidence: 0.9, hasConf: true,
		},
		{
			name: "explicit marker beyond window",
			messages: conversation(unrelatedHistory(10),
				user("Unrelated question: how do I renew a passport?")),
			class: ClassUnknown, reason: ReasonHistoryBeyondWindow,
		},
		{
			name:     "marker plus carry-over",
			messages: conversation(auth, user("Different topic, but in `auth.ts` the token check fails")),
			class:    ClassUnknown, reason: ReasonConflicting,
		},
		{
			name:     "acknowledgement",
			messages: conversation(refactor, user("thanks!")),
			class:    ClassContinuation, reason: ReasonAcknowledgement, confidence: 0.5, hasConf: true,
		},
		{
			name:     "acknowledgement word inside a request",
			messages: conversation(refactor, user("thanks, now unrelated work: book a flight to Oslo")),
			class:    ClassChange, reason: ReasonExplicitChange,
		},
		{
			name:     "too short",
			messages: conversation(refactor, user("why?")),
			class:    ClassUnknown, reason: ReasonInsufficientText,
		},
		{
			name:     "final assistant reply",
			messages: conversation(refactor),
			class:    ClassUnknown, reason: ReasonNoLiveUserTurn, coverage: CoveragePartial,
		},
		{
			name:     "final tool-call-only assistant",
			messages: conversation(refactor, user("list files"), assistantCalls(toolCall("t1", "ls", ""))),
			class:    ClassUnknown, reason: ReasonNoLiveUserTurn,
		},
		{
			name:     "empty assistant prefill",
			messages: conversation(refactor, user("continue"), llmprotocol.Message{Role: llmprotocol.RoleAssistant}),
			class:    ClassUnknown, reason: ReasonNoLiveUserTurn,
		},
		{
			name:     "first turn",
			messages: conversation(user("How do I configure the router for two backends?")),
			class:    ClassUnknown, reason: ReasonNoPriorTurn,
		},
		{
			name: "opaque-only live turn",
			messages: conversation(refactor,
				llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{image()}}),
			class: ClassUnknown, reason: ReasonOpaqueOnly,
		},
		{
			name: "CJK continuation",
			messages: conversation(exchange("我们讨论数据库索引优化方案", "可以先分析慢查询日志"),
				user("数据库索引应该怎么优化")),
			class: ClassContinuation, reason: ReasonLexicalOverlap,
		},
		{
			name: "CJK change",
			messages: conversation(exchange("我们讨论数据库索引优化方案", "可以先分析慢查询日志"),
				user("今天天气很好适合去公园散步")),
			class: ClassChange, reason: ReasonDisjoint,
		},
		{
			name: "clean plus ambiguous phrase",
			messages: conversation(refactor, user("Unrelated question: how do I renew a passport? "+
				"This is not unrelated to the earlier travel rule.")),
			class: ClassContinuation, reason: ReasonReference,
		},
		{
			name:     "clean only",
			messages: conversation(refactor, user("New topic: what is a good visa option for Japan?")),
			class:    ClassChange, reason: ReasonExplicitChange,
		},
		{
			name: "ambiguous only",
			messages: conversation(refactor,
				user("Can you check the retry logic, or is this a new topic for later")),
			class: ClassUnknown, reason: ReasonAmbiguousMarker,
		},
		{
			name:     "single-quoted marker with a reference",
			messages: conversation(refactor, user("'new topic' is the heading used in the document above")),
			class:    ClassContinuation, reason: ReasonReference,
		},
		{
			name:     "curly single-quoted marker",
			messages: conversation(refactor, user("‘unrelated’ was the label we picked for the bucket")),
			class:    ClassUnknown, reason: ReasonAmbiguousMarker,
		},
		{
			name:     "apostrophes are not quotes",
			messages: conversation(refactor, user("New topic: what's the user's best option for a visa?")),
			class:    ClassChange, reason: ReasonExplicitChange,
		},
		{
			name:     "unclosed delimiter hides nothing for continuation",
			messages: conversation(refactor, user(`Fix it "as you said above, then rename the module`)),
			class:    ClassContinuation, reason: ReasonReference,
		},
		{
			name: "masked span is a phrase barrier",
			messages: conversation(refactor,
				user(`new "quoted material" topic about capital cities of Peru and Chile`)),
			class: ClassChange, reason: ReasonDisjoint,
		},
		{
			name:     "phrase inside one run after a mask",
			messages: conversation(refactor, user("`code` new topic: what's a good visa?")),
			class:    ClassChange, reason: ReasonExplicitChange,
		},
		{
			name:     "negation lookback across a barrier",
			messages: conversation(refactor, user("not `x` unrelated to the capital of Peru and Chile")),
			class:    ClassUnknown, reason: ReasonAmbiguousMarker,
		},
	}
}

func TestGoldenFixtures(t *testing.T) {
	for _, tc := range goldenCases() {
		t.Run(tc.name, func(t *testing.T) {
			policy := defaultPolicy
			if tc.policy != nil {
				policy = *tc.policy
			}
			result := evaluate(tc.messages, policy)
			assertInvariants(t, result)
			if result.Class != tc.class || result.Reason != tc.reason {
				t.Fatalf("got %s/%s (conf %.3f, coverage %s, features %+v), want %s/%s",
					result.Class, result.Reason, result.Confidence, result.Coverage, result.Features,
					tc.class, tc.reason)
			}
			if tc.hasConf && !near(result.Confidence, tc.confidence) {
				t.Fatalf("confidence = %v, want %v", result.Confidence, tc.confidence)
			}
			if tc.coverage != "" && result.Coverage != tc.coverage {
				t.Fatalf("coverage = %s, want %s", result.Coverage, tc.coverage)
			}
		})
	}
}

func TestMarkerFlagsAreMutuallyExclusive(t *testing.T) {
	for _, tc := range goldenCases() {
		policy := defaultPolicy
		if tc.policy != nil {
			policy = *tc.policy
		}
		result := evaluate(tc.messages, policy)
		if result.Features.ChangeMarker && result.Features.MarkerAmbiguous {
			t.Fatalf("%s: both marker flags set", tc.name)
		}
	}
}

func TestOffsetsSurviveUnicodeAndMasking(t *testing.T) {
	// İ lowercases to a longer byte sequence; masked spans precede the quote.
	live := "İ ```code``` \"İx\" 'new topic' here"
	tokens := changeView(textSegment(live))
	spans := singleQuoteSpans(live)
	found := false
	for _, token := range tokens {
		if token.Text == "new" {
			found = true
			if !insideAny(spans, token.Offset) {
				t.Fatalf("token offset %d not inside quote spans %v", token.Offset, spans)
			}
			if !strings.HasPrefix(live[token.Offset:], "new") {
				t.Fatalf("offset %d does not point at the original token", token.Offset)
			}
		}
	}
	if !found {
		t.Fatal("token not found")
	}
	clean, ambiguous := changeMarkers(context.Background(), []textSegment{textSegment(live)})
	if clean || !ambiguous {
		t.Fatalf("clean=%v ambiguous=%v, want ambiguous only", clean, ambiguous)
	}
}

func TestUserOnlyPolicyScope(t *testing.T) {
	policy := defaultPolicy
	policy.IncludeAssistant = false
	result := evaluate(conversation(exchange("question about parsers", "answer about parsers"),
		user("parsers again please")), policy)
	if result.Scope.AssistantIncluded {
		t.Fatal("AssistantIncluded should be false")
	}
	prepared := prepare(context.Background(), conversation(exchange("q", "assistant text"), user("live")), true, policy)
	if len(prepared.Prior) != 1 || len(prepared.Prior[0].Assistant) != 0 {
		t.Fatalf("assistant evidence leaked: %+v", prepared.Prior)
	}
}

func TestExcludedContentScope(t *testing.T) {
	messages := conversation(user("Find the config"),
		assistantCalls(toolCall("t1", "read_file", `{"path":"x"}`)), toolResult("t1", "routing.yaml contents"),
		assistant("Found it."), user("Explain the routing.yaml section"))
	result := evaluate(messages, defaultPolicy)
	if !result.Scope.ExcludedContentPresent {
		t.Fatalf("expected ExcludedContentPresent, got %+v", result.Scope)
	}
}

func TestZeroOverlapParaphraseIsNeverHighConfidenceChange(t *testing.T) {
	result := evaluate(conversation(
		exchange("Our checkout page converts poorly on mobile", "Consider simplifying the form."),
		user("Would reducing required fields improve completion rates noticeably?")), defaultPolicy)
	assertInvariants(t, result)
	if result.Class == ClassChange && (result.Reason != ReasonDisjoint || result.Confidence > 0.6) {
		t.Fatalf("zero-overlap paraphrase gave %s/%s %.2f", result.Class, result.Reason, result.Confidence)
	}
}

func TestLongCodeReplyIsNotTruncatedUnderDefaults(t *testing.T) {
	var code strings.Builder
	for i := 0; code.Len() < 12*1024; i++ {
		fmt.Fprintf(&code, "func handler%d(ctx context.Context) error { return nil }\n", i)
	}
	result := evaluate(conversation(exchange("Write the handlers", code.String()),
		user("Now add logging to handler3")), defaultPolicy)
	if result.Coverage != CoverageFull || result.Scope.FeatureCapReached {
		t.Fatalf("coverage = %s scope = %+v", result.Coverage, result.Scope)
	}
}

func TestLongDelimitedEntityIsCapturedWhole(t *testing.T) {
	long := strings.Repeat("abcdefghij", 30)
	for _, quoted := range []string{"`" + long + "`", `"` + long + `"`} {
		entities, capped := segmentEntities(textSegment("see " + quoted + " now"))
		if capped {
			t.Fatal("unexpected cap")
		}
		if _, ok := entities[long[:maxEntityBytes]]; !ok {
			t.Fatalf("long span not captured as a prefix: %v", entities)
		}
	}
	result := evaluate(conversation(exchange("Why does `"+long+"` fail on startup", "Unknown."),
		user("Explain `"+long+"` once more")), defaultPolicy)
	if result.Class != ClassContinuation {
		t.Fatalf("got %s/%s", result.Class, result.Reason)
	}
}
