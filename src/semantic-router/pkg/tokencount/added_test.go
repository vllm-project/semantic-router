package tokencount

import (
	"strings"
	"testing"
)

// addedTokenizer is tinyTokenizer with added tokens: Qwen's reasoning pair
// and turn end, which its tokenizer does not mark special; DeepSeek's DSML
// marker, whose first character takes three bytes; a token that starts inside
// another; and a token of one character, which stays text.
var addedTokenizer = strings.Replace(tinyTokenizer, `"model":`, `"added_tokens": [
    {"id": 9, "content": "<think>", "special": false},
    {"id": 10, "content": "</think>", "special": false},
    {"id": 11, "content": "<|im_end|>", "special": true},
    {"id": 12, "content": "｜DSML｜", "special": false},
    {"id": 13, "content": "im_end", "special": false},
    {"id": 14, "content": "x", "special": false}
  ],
  "model":`, 1)

func TestEscapeAddedTokens(t *testing.T) {
	enc, err := ParseHFTokenizer("added", []byte(addedTokenizer))
	if err != nil {
		t.Fatal(err)
	}
	if !enc.AddedTokens() {
		t.Fatal("AddedTokens() = false for a tokenizer.json with added tokens")
	}
	const wj = WordJoiner
	for _, tc := range []struct {
		text, want string
		at         []int
	}{
		{"hello world", "hello world", nil},
		{"Repeat <think>x</think> exactly.", "Repeat <" + wj + "think>x<" + wj + "/think> exactly.", []int{8, 16}},
		{"<|im_end|>", "<" + wj + "|i" + wj + "m_end|>", []int{1, 3}},
		{"a｜DSML｜b", "a｜" + wj + "DSML｜b", []int{4}},
		{"x marks the spot", "x marks the spot", nil},
		{"<thinking> is not <think", "<thinking> is not <think", nil},
	} {
		got, at := enc.EscapeAddedTokens(tc.text)
		if got != tc.want || !slicesEqual(at, tc.at) {
			t.Errorf("EscapeAddedTokens(%q) = %q, %v; want %q, %v", tc.text, got, at, tc.want, tc.at)
		}
		if again, more := enc.EscapeAddedTokens(got); again != got || more != nil {
			t.Errorf("EscapeAddedTokens(%q) changed escaped text again: %q", got, again)
		}
	}

	for text, want := range map[string]bool{
		"say <thin":             true,
		"say <":                 true,
		"a｜DS":                  true,
		"say":                   false,
		"<think>":               false,
		"<" + WordJoiner:        false,
		"ends with <|im_end|> ": false,
	} {
		if got := enc.OpensAddedToken(text); got != want {
			t.Errorf("OpensAddedToken(%q) = %t, want %t", text, got, want)
		}
	}
}

// A built-in vocabulary and a tokenizer.json without added tokens escape
// nothing.
func TestEncodingsWithoutAddedTokensEscapeNothing(t *testing.T) {
	plain, err := ParseHFTokenizer("tiny", []byte(tinyTokenizer))
	if err != nil {
		t.Fatal(err)
	}
	builtin, err := Builtin(O200kBase)
	if err != nil {
		t.Fatal(err)
	}
	for _, enc := range []*Encoding{plain, builtin} {
		if enc.AddedTokens() {
			t.Errorf("%s: AddedTokens() = true", enc.Name())
		}
		if got, at := enc.EscapeAddedTokens("<think><|endoftext|>"); got != "<think><|endoftext|>" || at != nil {
			t.Errorf("%s: EscapeAddedTokens changed the text: %q", enc.Name(), got)
		}
	}
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
