package testcases

import (
	"strings"
	"testing"
)

func TestSplitBodyKeepsBytesInOrder(t *testing.T) {
	body := []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`)
	for _, writes := range []int{1, 3, 7, len(body), len(body) + 5} {
		pieces := splitBody(body, writes)
		if got := strings.Join(pieces, ""); got != string(body) {
			t.Fatalf("writes=%d: joined pieces %q, want %q", writes, got, body)
		}
		want := writes
		if want > len(body) {
			want = len(body)
		}
		if len(pieces) != want {
			t.Fatalf("writes=%d: got %d pieces, want %d", writes, len(pieces), want)
		}
		for i, piece := range pieces {
			if piece == "" {
				t.Fatalf("writes=%d: piece %d is empty", writes, i)
			}
		}
	}
}
