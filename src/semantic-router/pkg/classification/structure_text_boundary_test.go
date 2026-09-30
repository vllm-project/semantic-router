package classification

import (
	"fmt"
	"strings"
	"testing"
)

func TestKeywordOccurrenceBoundaries(t *testing.T) {
	for _, test := range []struct {
		text string
		want int
	}{
		{"must", 1},
		{"must must", 2},
		{"mustard almostmust must2", 0},
		{"émust musté ٣must", 0},
		{"必须must严格", 1},
		{"🙂must🙂", 1},
		{"\xffmust\xff", 1},
		{"must\x00must", 2},
	} {
		t.Run(fmt.Sprintf("%q", test.text), func(t *testing.T) {
			if got := keywordOccurrenceCount(test.text, []string{"must"}, false); got != test.want {
				t.Fatalf("keyword count = %d, want %d", got, test.want)
			}
		})
	}
}

// The built-in balance long-context probes repeatedly contain "without".
// Decoding the preceding rune must not rescan the entire growing prefix.
func BenchmarkKeywordOccurrenceLongContext(b *testing.B) {
	for _, repetitions := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("repetitions=%d", repetitions), func(b *testing.B) {
			text := strings.Repeat("Neutral long-context payload without routing intent. ", repetitions)
			b.SetBytes(int64(len(text)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if got := keywordOccurrenceCount(text, []string{"without", "must", "exactly"}, false); got != repetitions {
					b.Fatalf("keyword count = %d, want %d", got, repetitions)
				}
			}
		})
	}
}
