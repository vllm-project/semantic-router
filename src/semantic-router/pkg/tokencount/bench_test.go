package tokencount

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// benchmarkText is size bytes of the file TOKENCOUNT_BENCH_CORPUS names,
// repeated, or else of prose over a Zipf-distributed vocabulary of made-up
// words, with numbers, punctuation and line breaks: harder for the piece
// cache than real English, whose common words repeat more.
func benchmarkText(size int) string {
	if path := os.Getenv("TOKENCOUNT_BENCH_CORPUS"); path != "" {
		corpus, err := os.ReadFile(path)
		if err == nil && len(corpus) > 0 {
			text := strings.Repeat(string(corpus), size/len(corpus)+1)[:size]
			return strings.ToValidUTF8(text, "")
		}
	}
	rng := rand.New(rand.NewSource(int64(size)))
	syllables := []string{"ka", "lo", "mi", "ne", "tor", "vex", "an", "qui", "zel", "dra", "pho", "ri", "su", "gen", "ul", "bra"}
	const vocabulary = 50_000
	words := make([]string, vocabulary)
	for i := range words {
		var w strings.Builder
		for n := 1 + rng.Intn(4); n > 0; n-- {
			w.WriteString(syllables[rng.Intn(len(syllables))])
		}
		words[i] = w.String()
	}
	zipf := rand.NewZipf(rng, 1.1, 1, vocabulary-1)
	var b strings.Builder
	b.Grow(size + 64)
	for b.Len() < size {
		switch r := rng.Intn(100); {
		case r < 3:
			fmt.Fprintf(&b, " %d", rng.Intn(100_000))
		case r < 8:
			b.WriteString(",")
		case r < 11:
			b.WriteString(".\n")
		default:
			b.WriteString(" ")
			b.WriteString(words[zipf.Uint64()])
		}
	}
	return b.String()[:size]
}

func reportLatency(b *testing.B, samples []time.Duration) {
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	quantile := func(q float64) float64 {
		return float64(samples[min(len(samples)-1, int(q*float64(len(samples))))].Microseconds()) / 1000
	}
	b.ReportMetric(quantile(0.50), "p50-ms")
	b.ReportMetric(quantile(0.99), "p99-ms")
}

// benchmarkCount counts each size on 1, 2, 4 and 8 cores.
func benchmarkCount(b *testing.B, enc *Encoding) {
	for _, size := range []int{256 << 10, 1 << 20, 4 << 20, 8 << 20} {
		text := benchmarkText(size)
		for _, cores := range []int{1, 2, 4, 8} {
			b.Run(fmt.Sprintf("%dKiB/cores=%d", size>>10, cores), func(b *testing.B) {
				defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(cores))
				b.SetBytes(int64(size))
				samples := make([]time.Duration, 0, b.N)
				tokens := 0
				for b.Loop() {
					started := time.Now()
					tokens = enc.Count(text)
					samples = append(samples, time.Since(started))
				}
				b.ReportMetric(float64(tokens), "tokens")
				reportLatency(b, samples)
			})
		}
	}
}

func BenchmarkCountO200kBase(b *testing.B) {
	enc, err := Builtin(O200kBase)
	if err != nil {
		b.Fatal(err)
	}
	benchmarkCount(b, enc)
}

// BenchmarkCountTokenizerJSON counts with the tokenizer.json that
// TOKENCOUNT_BENCH_TOKENIZER names.
func BenchmarkCountTokenizerJSON(b *testing.B) {
	path := os.Getenv("TOKENCOUNT_BENCH_TOKENIZER")
	if path == "" {
		b.Skip("TOKENCOUNT_BENCH_TOKENIZER names no tokenizer.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	enc, err := ParseHFTokenizer(path, data)
	if err != nil {
		b.Fatal(err)
	}
	benchmarkCount(b, enc)
}
