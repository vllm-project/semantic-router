package memory

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	bostonTurn    = formatTurnChunk("I live in Boston, near the Charles River.", "Got it, you live in Boston.")
	denverTurn    = formatTurnChunk("I just moved to Denver, and I live there now.", "Welcome to Denver!")
	nurseTurn     = formatTurnChunk("I work as a nurse at the children's hospital.", "Thanks, I'll remember you're a nurse.")
	paramedicTurn = formatTurnChunk("I changed jobs and now work as a paramedic.", "Congratulations on the paramedic job!")
	budget4kTurn  = formatTurnChunk("My budget for the Japan trip is $4,000.", "I'll plan the Japan trip around $4,000.")
	budget6kTurn  = formatTurnChunk("I raised my budget for the Japan trip to $6,000.", "Updated, the Japan trip budget is $6,000.")
	dogTurn       = formatTurnChunk("My dog is a beagle named Biscuit.", "Biscuit the beagle, noted.")
	birthdayTurn  = formatTurnChunk("Biscuit turned three today, so I bought my dog a new toy.", "Happy birthday to Biscuit!")
	peanutsTurn   = formatTurnChunk("I'm allergic to peanuts, so keep them out of any recipe.", "Understood, no peanuts.")
	notNurseTurn  = formatTurnChunk("I no longer work as a nurse.", "Understood.")
	leftCityTurn  = formatTurnChunk("I don't live in Boston anymore.", "Thanks for letting me know.")
	chicagoTurn   = formatTurnChunk("I moved to Chicago, and I live there now.", "Welcome to Chicago!")
	jobAndCity    = formatTurnChunk("I changed jobs and now work as a paramedic, and I live in Boston.", "Noted.")
	jobNearPark   = formatTurnChunk("I changed jobs and now work as a paramedic near Central Park.", "Congratulations!")
	movedNearPark = formatTurnChunk("I moved near Central Park.", "Nice neighborhood.")
	// Long enough that the correction is a near-duplicate for the default dedup threshold.
	hospitalTurn   = formatTurnChunk("I work as a nurse at the children's hospital near the old park on Main Street in Boston, next to the big library.", "")
	noHospitalTurn = formatTurnChunk("I no longer work as a nurse at the children's hospital near the old park on Main Street in Boston, next to the big library.", "")
)

type datedContent struct {
	content string
	daysAgo int
	undated bool
}

func sessionChunkOf(turns ...string) string {
	return strings.Join(turns, sessionTurnSeparator)
}

// injectedContents runs retrieved memories through the default memory filter.
func injectedContents(t *testing.T, retrieved []datedContent) []string {
	t.Helper()
	now := time.Now()
	results := make([]*RetrieveResult, 0, len(retrieved))
	for i, r := range retrieved {
		mem := &Memory{ID: fmt.Sprintf("m%d", i), Content: r.content}
		if !r.undated {
			mem.CreatedAt = now.AddDate(0, 0, -r.daysAgo)
		}
		results = append(results, &RetrieveResult{Memory: mem, Score: 0.5})
	}
	gate := NewReflectionGate(config.MemoryReflectionConfig{}, nil)
	require.NotNil(t, gate)
	injected := make([]string, 0, len(results))
	for _, r := range gate.Filter(results) {
		injected = append(injected, r.Memory.Content)
	}
	return injected
}

func TestReflectionGateDropsCorrectedTurns(t *testing.T) {
	cases := []struct {
		name      string
		retrieved []datedContent
		want      []string
	}{
		{
			name:      "a move hides the old city",
			retrieved: []datedContent{{content: bostonTurn, daysAgo: 30}, {content: denverTurn, daysAgo: 9}},
			want:      []string{denverTurn},
		},
		{
			name:      "a job change hides the old job",
			retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: paramedicTurn, daysAgo: 9}},
			want:      []string{paramedicTurn},
		},
		{
			name:      "a raised budget hides the old budget",
			retrieved: []datedContent{{content: budget4kTurn, daysAgo: 30}, {content: budget6kTurn, daysAgo: 9}},
			want:      []string{budget6kTurn},
		},
		{
			name: "no longer and anymore end an old fact",
			retrieved: []datedContent{
				{content: nurseTurn, daysAgo: 30},
				{content: bostonTurn, daysAgo: 30},
				{content: notNurseTurn, daysAgo: 9},
				{content: leftCityTurn, daysAgo: 9},
			},
			want: []string{notNurseTurn, leftCityTurn},
		},
		{
			name: "a correction keeps unrelated memories",
			retrieved: []datedContent{
				{content: dogTurn, daysAgo: 30},
				{content: peanutsTurn, daysAgo: 28},
				{content: bostonTurn, daysAgo: 30},
				{content: denverTurn, daysAgo: 9},
			},
			want: []string{dogTurn, peanutsTurn, denverTurn},
		},
		{
			name:      "a session chunk keeps its other turns",
			retrieved: []datedContent{{content: sessionChunkOf(dogTurn, bostonTurn, nurseTurn), daysAgo: 30}, {content: denverTurn, daysAgo: 9}},
			want:      []string{sessionChunkOf(dogTurn, nurseTurn), denverTurn},
		},
		{
			name: "a trimmed session chunk that repeats a turn chunk is deduplicated",
			retrieved: []datedContent{
				{content: sessionChunkOf(dogTurn, bostonTurn), daysAgo: 30},
				{content: dogTurn, daysAgo: 30},
				{content: denverTurn, daysAgo: 9},
			},
			want: []string{dogTurn, denverTurn},
		},
		{
			name:      "a later turn in one session chunk corrects an earlier one",
			retrieved: []datedContent{{content: sessionChunkOf(bostonTurn, dogTurn, denverTurn), daysAgo: 9}},
			want:      []string{sessionChunkOf(dogTurn, denverTurn)},
		},
		{
			name: "a correction of a correction leaves only the newest",
			retrieved: []datedContent{
				{content: bostonTurn, daysAgo: 30},
				{content: chicagoTurn, daysAgo: 20},
				{content: denverTurn, daysAgo: 9},
			},
			want: []string{denverTurn},
		},
		{
			name:      "a correction that holds another fact stays and still corrects",
			retrieved: []datedContent{{content: sessionChunkOf(nurseTurn, jobAndCity, dogTurn), daysAgo: 30}, {content: denverTurn, daysAgo: 9}},
			want:      []string{sessionChunkOf(jobAndCity, dogTurn), denverTurn},
		},
		{
			name: "a correction stays when its own correction skips what it corrected",
			retrieved: []datedContent{
				{content: nurseTurn, daysAgo: 30},
				{content: jobNearPark, daysAgo: 20},
				{content: movedNearPark, daysAgo: 9},
			},
			want: []string{jobNearPark, movedNearPark},
		},
		{
			name:      "the last turn of a newer session chunk corrects another memory",
			retrieved: []datedContent{{content: bostonTurn, daysAgo: 30}, {content: sessionChunkOf(dogTurn, denverTurn), daysAgo: 9}},
			want:      []string{sessionChunkOf(dogTurn, denverTurn)},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.ElementsMatch(t, tc.want, injectedContents(t, tc.retrieved))
		})
	}
}

func TestReflectionGateKeepsTurnsWithoutACorrection(t *testing.T) {
	apartment := formatTurnChunk("I rent an apartment in Boston.", "Noted, an apartment in Boston.")
	planned := formatTurnChunk("Next year I will live in Denver.", "Denver next year, noted.")
	workout := formatTurnChunk("I work out every morning.", "Morning workouts, noted.")
	park := formatTurnChunk("I live near Washington Park.", "That's a lovely park.")
	replicas := formatTurnChunk("My deployment runs three replicas.", "Three replicas, noted.")
	manifest := formatTurnChunk("Here is my manifest:\nkind: Service\n---\nkind: Deployment",
		"I changed my deployment to run five replicas.")
	partner := formatTurnChunk("My partner moved to Denver and started work as a teacher.", "Big change for your partner!")
	stayed := formatTurnChunk("I haven't moved, I still live in Boston.", "Good to know.")
	hypothetical := formatTurnChunk("If I switched jobs, I would still work as a nurse.", "That makes sense.")
	otherSentence := formatTurnChunk("I moved to Denver. My sister works as a nurse there.", "Nice to be near her.")
	bostonMove := formatTurnChunk("I moved to Boston, and I live there now.", "Welcome to Boston!")
	question := formatTurnChunk("Have I changed jobs, or do I still work as a nurse?", "You still work as a nurse.")
	inverted := formatTurnChunk("Had I switched jobs, I would still work as a nurse.", "That makes sense.")
	otherClause := formatTurnChunk("I moved apartments, and I still work as a nurse at another clinic.", "Congrats on the new place.")
	nowClause := formatTurnChunk("I moved apartments, and my dog is now three years old.", "Happy birthday to your dog!")
	dogName := formatTurnChunk("My dog Biscuit is a beagle.", "A beagle named Biscuit, noted.")
	nowName := formatTurnChunk("I moved apartments, and now Biscuit is three.", "Happy birthday, Biscuit!")
	twoFacts := formatTurnChunk("I live in Boston and work as a nurse.", "Noted.")
	twoSentences := formatTurnChunk("I live in Boston. I work as a nurse.", "Noted.")
	commaSplice := formatTurnChunk("I live in Boston, I work as a nurse.", "Noted.")
	someday := formatTurnChunk("If someday I switched jobs to work as a paramedic, I'd tell you.", "Please do.")
	commute := formatTurnChunk("My work is in Cambridge, so I commute.", "That's a long ride.")
	closerWork := formatTurnChunk("I moved apartments, and now work is closer.", "Nice, a shorter commute.")

	cases := []struct {
		name      string
		retrieved []datedContent
	}{
		{name: "a newer fact about the same dog", retrieved: []datedContent{{content: dogTurn, daysAgo: 30}, {content: birthdayTurn, daysAgo: 21}}},
		{name: "a second fact about the same city", retrieved: []datedContent{{content: bostonTurn, daysAgo: 30}, {content: apartment, daysAgo: 9}}},
		{name: "a planned move", retrieved: []datedContent{{content: bostonTurn, daysAgo: 30}, {content: planned, daysAgo: 9}}},
		{name: "a shared word without a shared pair", retrieved: []datedContent{{content: workout, daysAgo: 30}, {content: paramedicTurn, daysAgo: 9}}},
		{name: "a correction older than the statement", retrieved: []datedContent{{content: denverTurn, daysAgo: 30}, {content: park, daysAgo: 9}}},
		{name: "undated memories", retrieved: []datedContent{{content: bostonTurn, undated: true}, {content: denverTurn, undated: true}}},
		{name: "an assistant reply after a --- line", retrieved: []datedContent{{content: replicas, daysAgo: 30}, {content: manifest, daysAgo: 9}}},
		{
			name:      "a session chunk's copy of a correction",
			retrieved: []datedContent{{content: denverTurn, daysAgo: 9}, {content: sessionChunkOf(denverTurn, paramedicTurn), daysAgo: 8}},
		},
		{name: "a change someone else made", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: partner, daysAgo: 9}}},
		{name: "a change the user denies", retrieved: []datedContent{{content: bostonTurn, daysAgo: 30}, {content: stayed, daysAgo: 9}}},
		{name: "a hypothetical change", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: hypothetical, daysAgo: 9}}},
		{name: "a change in another sentence", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: otherSentence, daysAgo: 9}}},
		{
			name:      "an earlier turn in a newer session chunk",
			retrieved: []datedContent{{content: denverTurn, daysAgo: 20}, {content: sessionChunkOf(bostonMove, dogTurn), daysAgo: 10}},
		},
		{name: "a question about a change", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: question, daysAgo: 9}}},
		{name: "an inverted hypothetical", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: inverted, daysAgo: 9}}},
		{name: "a change in another clause", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: otherClause, daysAgo: 9}}},
		{name: "a now clause about something else", retrieved: []datedContent{{content: dogTurn, daysAgo: 30}, {content: nowClause, daysAgo: 9}}},
		{name: "a now clause about someone else", retrieved: []datedContent{{content: dogName, daysAgo: 30}, {content: nowName, daysAgo: 9}}},
		{name: "two facts joined by and", retrieved: []datedContent{{content: twoFacts, daysAgo: 30}, {content: denverTurn, daysAgo: 9}}},
		{name: "two facts in two sentences", retrieved: []datedContent{{content: twoSentences, daysAgo: 30}, {content: denverTurn, daysAgo: 9}}},
		{name: "two facts in a comma splice", retrieved: []datedContent{{content: commaSplice, daysAgo: 30}, {content: denverTurn, daysAgo: 9}}},
		{name: "a hypothetical opened earlier in the clause", retrieved: []datedContent{{content: nurseTurn, daysAgo: 30}, {content: someday, daysAgo: 9}}},
		{name: "a noun after now", retrieved: []datedContent{{content: commute, daysAgo: 30}, {content: closerWork, daysAgo: 9}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := make([]string, 0, len(tc.retrieved))
			for _, r := range tc.retrieved {
				want = append(want, r.content)
			}
			assert.ElementsMatch(t, want, injectedContents(t, tc.retrieved))
		})
	}
}

func TestReflectionGateTrimsACopyOfTheStoredSessionChunk(t *testing.T) {
	now := time.Now()
	chunk := sessionChunkOf(dogTurn, bostonTurn)
	stored := &Memory{ID: "window", Content: chunk, CreatedAt: now.AddDate(0, 0, -30)}
	correction := &Memory{ID: "move", Content: denverTurn, CreatedAt: now.AddDate(0, 0, -9)}

	gate := NewReflectionGate(config.MemoryReflectionConfig{}, nil)
	got := gate.Filter([]*RetrieveResult{{Memory: stored, Score: 0.5}, {Memory: correction, Score: 0.4}})

	require.Len(t, got, 2)
	assert.Equal(t, chunk, stored.Content, "the stored record must not change")
	for _, r := range got {
		assert.NotContains(t, r.Memory.Content, "Boston")
	}
}

func TestReflectionGateHidesAFactOnlyWhenItsCorrectionIsInjected(t *testing.T) {
	chunk := sessionChunkOf(dogTurn, bostonTurn, nurseTurn)
	cases := []struct {
		name      string
		maxTokens int
		want      []string
	}{
		{name: "both fit the token budget", maxTokens: 2048, want: []string{sessionChunkOf(dogTurn, nurseTurn), denverTurn}},
		{name: "the correction misses the token budget", maxTokens: estimateTokens(chunk), want: []string{chunk}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Now()
			gate := NewReflectionGate(config.MemoryReflectionConfig{MaxInjectTokens: tc.maxTokens}, nil)
			got := gate.Filter([]*RetrieveResult{
				{Memory: &Memory{ID: "window", Content: chunk, CreatedAt: now.AddDate(0, 0, -2)}, Score: 0.9},
				{Memory: &Memory{ID: "move", Content: denverTurn, CreatedAt: now.AddDate(0, 0, -1)}, Score: 0.3},
			})
			injected := make([]string, 0, len(got))
			for _, r := range got {
				injected = append(injected, r.Memory.Content)
			}
			assert.Equal(t, tc.want, injected)
		})
	}
}

func TestReflectionGateDedupKeepsACorrectionBesideTheFactItCorrects(t *testing.T) {
	other := formatTurnChunk("I'm planning a trip to Japan next spring and want to see Kyoto, Osaka and the temples.", "")
	require.GreaterOrEqual(t, wordJaccard(hospitalTurn, noHospitalTurn), float32(0.90), "plain dedup would drop the lower-scored correction")

	cases := []struct {
		name      string
		maxTokens int
		want      []string
	}{
		{name: "both fit the token budget", maxTokens: 2048, want: []string{"other", "correction"}},
		{name: "the correction misses the token budget", maxTokens: estimateTokens(other) + estimateTokens(hospitalTurn), want: []string{"other", "old"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Now()
			gate := NewReflectionGate(config.MemoryReflectionConfig{MaxInjectTokens: tc.maxTokens}, nil)
			got := gate.Filter([]*RetrieveResult{
				{Memory: &Memory{ID: "other", Content: other, CreatedAt: now}, Score: 0.95},
				{Memory: &Memory{ID: "old", Content: hospitalTurn, CreatedAt: now.AddDate(0, 0, -30)}, Score: 0.9},
				{Memory: &Memory{ID: "correction", Content: noHospitalTurn, CreatedAt: now.AddDate(0, 0, -9)}, Score: 0.5},
			})
			ids := make([]string, 0, len(got))
			for _, r := range got {
				ids = append(ids, r.Memory.ID)
			}
			assert.Equal(t, tc.want, ids)
		})
	}
}

func BenchmarkReflectionGateSupersedesLargeRepetitiveChunks(b *testing.B) {
	turn := formatTurnChunk("I changed "+strings.Repeat("alpha ", 500), "Noted.")
	chunk := sessionChunkOf(turn, turn, turn, turn, turn)
	gate := NewReflectionGate(config.MemoryReflectionConfig{MaxInjectTokens: 1 << 20}, nil)
	now := time.Now()
	for i := 0; i < b.N; i++ {
		retrieved := make([]*RetrieveResult, 0, 10)
		for m := 0; m < 10; m++ {
			retrieved = append(retrieved, &RetrieveResult{
				Memory: &Memory{ID: fmt.Sprint(m), Content: chunk, CreatedAt: now.Add(-time.Duration(m) * time.Hour)},
				Score:  0.5,
			})
		}
		gate.Filter(retrieved)
	}
}
