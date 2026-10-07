package parity

import (
	"bytes"
	"fmt"
	"strings"
)

// Diff returns "" when want and got encode identically, and otherwise a
// line diff of their JSON encodings ("-" want, "+" got).
func Diff(want, got *Record) string {
	wantJSON, err := want.Encode()
	if err != nil {
		return fmt.Sprintf("encode want: %v", err)
	}
	gotJSON, err := got.Encode()
	if err != nil {
		return fmt.Sprintf("encode got: %v", err)
	}
	return DiffText(wantJSON, gotJSON)
}

// DiffText line-diffs two encodings; it returns "" when they are equal.
func DiffText(want, got []byte) string {
	if bytes.Equal(want, got) {
		return ""
	}
	a := strings.Split(string(want), "\n")
	b := strings.Split(string(got), "\n")
	// Longest common subsequence table, filled from the end.
	lcs := make([][]int, len(a)+1)
	for i := range lcs {
		lcs[i] = make([]int, len(b)+1)
	}
	for i := len(a) - 1; i >= 0; i-- {
		for j := len(b) - 1; j >= 0; j-- {
			if a[i] == b[j] {
				lcs[i][j] = lcs[i+1][j+1] + 1
			} else {
				lcs[i][j] = max(lcs[i+1][j], lcs[i][j+1])
			}
		}
	}
	var out strings.Builder
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		switch {
		case i < len(a) && j < len(b) && a[i] == b[j]:
			i, j = i+1, j+1
		case j < len(b) && (i == len(a) || lcs[i][j+1] >= lcs[i+1][j]):
			fmt.Fprintf(&out, "+%d: %s\n", j+1, b[j])
			j++
		default:
			fmt.Fprintf(&out, "-%d: %s\n", i+1, a[i])
			i++
		}
	}
	return out.String()
}
