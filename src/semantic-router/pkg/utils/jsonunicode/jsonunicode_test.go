package jsonunicode

import (
	"testing"
	"unicode/utf8"
)

func TestValid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		raw  []byte
		want bool
	}{
		{name: "empty is Unicode-valid but not JSON", raw: nil, want: true},
		{name: "ASCII JSON", raw: []byte(`{"value":"text"}`), want: true},
		{name: "literal Unicode", raw: []byte(`{"value":"你好"}`), want: true},
		{name: "literal replacement character", raw: []byte(`{"value":"�"}`), want: true},
		{name: "escaped replacement character", raw: []byte(`{"value":"\ufffd"}`), want: true},
		{name: "lowercase surrogate pair", raw: []byte(`{"value":"\ud83d\ude00"}`), want: true},
		{name: "uppercase surrogate pair", raw: []byte(`{"value":"\uD83D\uDE00"}`), want: true},
		{name: "escaped backslash before surrogate text", raw: []byte(`{"value":"\\ud800"}`), want: true},
		{name: "invalid UTF-8 in string", raw: []byte{'{', '"', 'v', '"', ':', '"', 0xff, '"', '}'}},
		{name: "invalid UTF-8 outside string", raw: []byte{0xff, '{', '}'}},
		{name: "unpaired high surrogate", raw: []byte(`{"value":"\ud800"}`)},
		{name: "unpaired low surrogate", raw: []byte(`{"value":"\udc00"}`)},
		{name: "reversed surrogate pair", raw: []byte(`{"value":"\udc00\ud800"}`)},
		{name: "high surrogate followed by scalar", raw: []byte(`{"value":"\ud800\u0041"}`)},
		{name: "high surrogate followed by escaped slash", raw: []byte(`{"value":"\ud800\\udc00"}`)},
		{name: "truncated Unicode escape", raw: []byte(`{"value":"\u12"}`)},
		{name: "non-hex Unicode escape", raw: []byte(`{"value":"\uZZZZ"}`)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := Valid(test.raw); got != test.want {
				t.Fatalf("Valid(%q) = %v, want %v", test.raw, got, test.want)
			}
		})
	}
}

func FuzzValidNeverAcceptsInvalidUTF8OrPanics(f *testing.F) {
	for _, seed := range [][]byte{
		nil,
		[]byte(`{"value":"\ud83d\ude00"}`),
		[]byte(`{"value":"\ud800"}`),
		{'{', '"', 0xff, '"', '}'},
		[]byte(`{"value":"\\\"\uD83D\uDE00"}`),
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		valid := Valid(raw)
		if valid && !utf8.Valid(raw) {
			t.Fatalf("Valid accepted invalid UTF-8: %q", raw)
		}
	})
}
