package topiccontinuity

import (
	"reflect"
	"testing"
	"unicode"
)

// TestTokenizerGoldens freezes the tokenizer's v1 behavior. A change here is
// an evaluator behavior change and must bump EvaluatorVersion.
func TestTokenizerGoldens(t *testing.T) {
	cases := map[string][]string{
		"Refactor the routing_module now": {"refactor", "the", "routing_module", "now"},
		"'parser' and parser":             {"parser", "and", "parser"},
		"isn't it":                        {"isn", "it"},
		"x + y = 3":                       {"+", "="},
		"auth.ts validateToken":           {"auth", "ts", "validatetoken"},
		"数据库索引":                           {"数", "据", "数据", "库", "据库", "索", "库索", "引", "索引"},
		"İstanbul":                        {"istanbul"},
		"":                                {},
	}
	for input, want := range cases {
		got := tokenizeTerms(input)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("tokenizeTerms(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestPhraseNormalizationTrimsApostrophesFromEdgesOnly(t *testing.T) {
	tokens := continuationView(textSegment("'new topic' isn't the user’s ’quote’"))
	var got []string
	for _, token := range tokens {
		got = append(got, token.Text)
	}
	want := []string{"new", "topic", "isn't", "the", "user’s", "quote"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %q, want %q", got, want)
	}
}

func TestCJKFastPathIsExact(t *testing.T) {
	for value := rune(0); value < 0x1100; value++ {
		if unicode.In(value, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
			t.Fatalf("U+%04X is CJK below the fast-path bound", value)
		}
	}
}

func TestEntityScannerClasses(t *testing.T) {
	cases := map[string][]string{
		"see `Some Span` here":                {"some span"},
		`say "hi" and "hello there"`:          {"hello there"},
		"open pkg/my_file.go now":             {"pkg/my_file.go", "my_file.go", "my_file"},
		"call auth.ts and validateToken":      {"auth.ts", "validatetoken"},
		"the JsonParser and httpServer types": {"jsonparser", "httpserver"},
		// Acronym-led names have no lower-to-upper transition, as in v1.
		"the HTTPServer and JSONParser types": {},
		"ids 12 345 and var_1100.":            {"345", "var_1100", "1100"},
		"plain words only":                    {},
	}
	for input, want := range cases {
		got, capped := segmentEntities(textSegment(input))
		if capped {
			t.Fatalf("%q: unexpected cap", input)
		}
		if len(got) != len(want) {
			t.Fatalf("%q: got %v, want %v", input, got, want)
		}
		for _, value := range want {
			if _, ok := got[value]; !ok {
				t.Fatalf("%q: missing %q in %v", input, value, got)
			}
		}
	}
}
