//go:build contextwindowmeasure

package extproc

import (
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// commonWords are English words that the proxy tokenizer reads as one token
// each, for prose of about five bytes a token.
var commonWords = strings.Fields(`the of and to in is was that for it with as on be at by this had not are
but from or have an they which one you were her all she there would their we him been has when who will more
no if out so said what up its about into than them can only other new some could time these two may then do
first any my now such like our over man me even most made after also did many before must through back years
where much your way well down should because each just those people how too little state good very make world
still own see men work long get here between both life being under never day same another know while last
might us great old year off come since against go came right used take three`)

// variedPrompt returns about tokens tokens of prose drawn from seed, with
// numbers, punctuation and line breaks, and its count: English words, or
// made-up words over a Zipf-distributed vocabulary, which share fewer pieces
// and so cost the count more. A request built from a seed no other request
// uses is answered neither from the guard's decision cache nor from pieces an
// earlier count has seen.
func variedPrompt(t *testing.T, seed int64, tokens int, english bool) (string, int) {
	t.Helper()
	proxy, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	rng := rand.New(rand.NewSource(seed))
	word := func() string { return commonWords[rng.Intn(len(commonWords))] }
	if !english {
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
		word = func() string { return words[zipf.Uint64()] }
	}
	var b strings.Builder
	write := func(size int) {
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
				b.WriteString(word())
			}
		}
	}
	write(1 << 16)
	write(int(float64(tokens) * float64(b.Len()) / float64(proxy.Count(b.String()))))
	text := b.String()
	return text, proxy.Count(text)
}

// TestMeasureContextWindowLatency measures what the window check adds to
// fitting requests, whose count runs beside a guard that answers each window
// in 50 ms, and how fast it refuses oversized ones. Run it with
// -tags contextwindowmeasure.
func TestMeasureContextWindowLatency(t *testing.T) {
	const runs = 30
	quantiles := func(samples []time.Duration) (time.Duration, time.Duration) {
		sorted := slices.Clone(samples)
		slices.Sort(sorted)
		return sorted[len(sorted)/2], sorted[min(len(sorted)-1, len(sorted)*99/100)]
	}
	ms := func(d time.Duration) string { return fmt.Sprintf("%.1f ms", float64(d.Microseconds())/1000) }
	checked, _ := slowGuardEngine(t, 50*time.Millisecond, true)
	unchecked, _ := slowGuardEngine(t, 50*time.Millisecond, false)
	refusing, guard := slowGuardEngine(t, 50*time.Millisecond, true)
	seed := int64(0)
	for _, english := range []bool{true, false} {
		kind := map[bool]string{true: "English", false: "made-up words"}[english]
		for _, target := range []int{256_000, 512_000, 1_000_000} {
			var with, without, waits, counts []time.Duration
			least, most, megabytes, uncounted := math.MaxInt, 0, 0.0, 0
			for range runs {
				seed++
				text, tokens := variedPrompt(t, seed, target, english)
				least, most, megabytes = min(least, tokens), max(most, tokens), float64(len(text))/(1<<20)
				logs := observeRouterEvents(t)
				plan, elapsed := planChat(t, checked, fmt.Sprintf("checked-%d", seed), text)
				require.NotNil(t, plan.Call)
				with = append(with, elapsed)
				if counted := logs.FilterMessage("context_window_counted").All(); len(counted) == 0 {
					uncounted++
					waits, counts = append(waits, 0), append(counts, 0)
				} else {
					fields := counted[0].ContextMap()
					require.Equal(t, false, fields["refused"])
					waits = append(waits, time.Duration(fields["wait_ms"].(float64)*float64(time.Millisecond)))
					counts = append(counts, time.Duration(fields["count_ms"].(int64))*time.Millisecond)
				}
				seed++
				text, _ = variedPrompt(t, seed, target, english)
				plan, elapsed = planChat(t, unchecked, fmt.Sprintf("unchecked-%d", seed), text)
				require.NotNil(t, plan.Call)
				without = append(without, elapsed)
			}
			w50, w99 := quantiles(waits)
			c50, c99 := quantiles(counts)
			on50, on99 := quantiles(with)
			off50, off99 := quantiles(without)
			t.Logf("fits %s, %d-%d tokens (%.1f MiB, %d of %d within the byte bound): waited p50 %s p99 %s for a count of p50 %s p99 %s; request p50 %s p99 %s with the check, p50 %s p99 %s without",
				kind, least, most, megabytes, uncounted, runs, ms(w50), ms(w99), ms(c50), ms(c99), ms(on50), ms(on99), ms(off50), ms(off99))
		}
		var refusals []time.Duration
		least, most, stray := math.MaxInt, 0, 0
		for range runs / 3 {
			seed++
			text, tokens := variedPrompt(t, seed, 1_150_000, english)
			require.Greater(t, tokens, 1_080_033, "the request must lie beyond the band")
			least, most = min(least, tokens), max(most, tokens)
			plan, elapsed := planChat(t, refusing, fmt.Sprintf("refused-%d", seed), text)
			require.NotNil(t, plan.Immediate)
			require.Equal(t, http.StatusBadRequest, plan.Immediate.Status)
			refusals = append(refusals, elapsed)
			sent := guard.windows()
			time.Sleep(300 * time.Millisecond)
			stray += guard.windows() - sent
		}
		r50, r99 := quantiles(refusals)
		t.Logf("refuses %s, %d-%d tokens, named tokenizer: 400 after p50 %s p99 %s; guard windows sent after the 400: %d",
			kind, least, most, ms(r50), ms(r99), stray)
	}
}
