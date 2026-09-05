package store

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSanitizePostgresTextStripsNULAndInvalidUTF8(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain ascii unchanged", "hello world", "hello world"},
		{"valid multibyte unchanged", "héllo — 世界", "héllo — 世界"},
		{"empty unchanged", "", ""},
		{"nul byte dropped", "a\x00b", "ab"},
		{"truncated utf8 replaced", "x\xef\xbfy", "x\uFFFDy"},
		{"lone continuation replaced", "a\x80b", "a\uFFFDb"},
		{"nul and invalid combined", "p\x00q\xffr", "pq\uFFFDr"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sanitizePostgresText(tc.in)
			if got != tc.want {
				t.Fatalf("sanitizePostgresText(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if strings.ContainsRune(got, 0) {
				t.Fatalf("sanitized text still contains a NUL byte: %q", got)
			}
			if !utf8.ValidString(got) {
				t.Fatalf("sanitized text is not valid UTF-8: %q", got)
			}
		})
	}
}

func TestSanitizePostgresJSONRemovesNULEscape(t *testing.T) {
	clean := []byte(`{"k":"value"}`)
	if got := sanitizePostgresJSON(clean); !bytes.Equal(got, clean) {
		t.Fatalf("clean JSON altered: got %q want %q", got, clean)
	}
	dirty := []byte(`{"k":"a\u0000b"}`)
	got := sanitizePostgresJSON(dirty)
	if bytes.Contains(got, []byte(`\u0000`)) {
		t.Fatalf("sanitized JSON still contains a NUL escape: %q", got)
	}
	if want := []byte(`{"k":"ab"}`); !bytes.Equal(got, want) {
		t.Fatalf("sanitizePostgresJSON = %q, want %q", got, want)
	}
}

// TestSanitizePostgresJSONPreservesLiteralEscape guards the case where a valid
// replay value's text literally contains the six characters \u0000. json.Marshal
// escapes the backslash, so the encoded bytes hold \\u0000 (an even backslash
// run) — that match is an escaped backslash, not a NUL escape, and must be kept.
// A blind replace would delete the second backslash, leave the first escaping
// the closing quote, and produce JSON that PostgreSQL rejects. A real NUL byte
// still marshals to \u0000 (an odd backslash run) and must be dropped.
func TestSanitizePostgresJSONPreservesLiteralEscape(t *testing.T) {
	cases := []struct {
		name string
		val  string
		want string // decoded value expected back after sanitize + Unmarshal
	}{
		{"literal backslash-u0000 preserved", `abc\u0000def`, `abc\u0000def`},
		{"real nul stripped", "a\x00b", "ab"},
		{"real nul next to literal escape", "a\x00b\\u0000c", `ab\u0000c`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := json.Marshal(map[string]string{"k": tc.val})
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			got := sanitizePostgresJSON(encoded)
			var back map[string]string
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatalf("sanitized JSON is invalid: %v (in=%q got=%q)", err, encoded, got)
			}
			if back["k"] != tc.want {
				t.Fatalf("value corrupted: got %q, want %q (sanitized=%q)", back["k"], tc.want, got)
			}
			if bytes.Contains(got, []byte("\\u0000\\u0000")) {
				t.Fatalf("unexpected doubled escape: %q", got)
			}
		})
	}
}

// TestPostgresInsertRecordSanitizesUntrustedText proves the store boundary no
// longer forwards Postgres-hostile bytes. A record whose body carries a NUL
// byte or truncated UTF-8 (as PDF-derived text does) previously failed at
// insert with "invalid byte sequence for encoding UTF8" / "unsupported Unicode
// escape sequence" and was silently dropped; after preparation the bound
// arguments are safe to store.
func TestPostgresInsertRecordSanitizesUntrustedText(t *testing.T) {
	built, err := newPostgresInsertRecord(Record{
		RequestBody:  "prompt\x00tail",
		ResponseBody: "resp\xef\xbfbody",
		SessionPolicy: map[string]interface{}{
			"note": "policy\x00value",
		},
	})
	if err != nil {
		t.Fatalf("newPostgresInsertRecord returned error: %v", err)
	}

	if got := built.record.RequestBody; strings.ContainsRune(got, 0) {
		t.Fatalf("request body still contains a NUL byte: %q", got)
	}
	if got := built.record.ResponseBody; !utf8.ValidString(got) {
		t.Fatalf("response body is not valid UTF-8: %q", got)
	}
	if bytes.Contains(built.sessionPolicyJSON, []byte(`\u0000`)) {
		t.Fatalf("session policy JSON still contains a NUL escape: %q", built.sessionPolicyJSON)
	}
}
