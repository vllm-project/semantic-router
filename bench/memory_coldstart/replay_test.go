package main

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

type gradeFlags struct {
	Hit, Right, Top1, Stale bool
}

func TestGrade(t *testing.T) {
	const (
		dog    = "Q: My dog is a beagle named Biscuit."
		boston = "Q: I live in Boston, near the Charles River."
		denver = "Q: I just moved to Denver, and I live there now."
	)
	dogName := probe{Phase: phaseFirstSeen, Query: "What is my dog's name?", Expect: []string{"biscuit"}}
	city := probe{Phase: phaseStale, Query: "Which city do I live in now?", Expect: []string{"denver"}, Stale: []string{"boston"}}
	unknown := probe{Phase: phaseNoMemory, Query: "When is my dentist appointment?"}

	cases := []struct {
		name     string
		probe    probe
		contents []string
		want     gradeFlags
	}{
		{name: "nothing retrieved", probe: dogName, want: gradeFlags{}},
		{name: "right memory ranked first", probe: dogName, contents: []string{dog, boston}, want: gradeFlags{Hit: true, Right: true, Top1: true}},
		{name: "right memory below a distractor", probe: dogName, contents: []string{boston, dog}, want: gradeFlags{Hit: true, Right: true}},
		{name: "stale memory outranks the correction", probe: city, contents: []string{boston, denver}, want: gradeFlags{Hit: true, Right: true, Stale: true}},
		{name: "only the stale memory", probe: city, contents: []string{boston}, want: gradeFlags{Hit: true, Stale: true}},
		{name: "hit with no memory to find", probe: unknown, contents: []string{dog}, want: gradeFlags{Hit: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			found := make([]*memory.RetrieveResult, 0, len(tc.contents))
			for _, content := range tc.contents {
				found = append(found, &memory.RetrieveResult{Memory: &memory.Memory{Content: content}, Score: 0.5})
			}
			got := grade(tc.probe, 0, found)
			assert.Equal(t, tc.want, gradeFlags{Hit: got.Hit, Right: got.Right, Top1: got.Top1, Stale: got.Stale})
			assert.Len(t, got.Retrieved, len(tc.contents))
		})
	}
}

func TestReplayBuiltinScenario(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "1")

	rep, err := replay(context.Background(), builtinScenario, replayOptions{Threshold: 0.10, Limit: 5})
	require.NoError(t, err)

	memoriesBeforeEachSession := make([]int, 0, len(rep.Sessions))
	for _, s := range rep.Sessions {
		memoriesBeforeEachSession = append(memoriesBeforeEachSession, s.Memories)
	}
	assert.Equal(t, []int{0, 5, 9, 11, 16}, memoriesBeforeEachSession)
	assert.Equal(t, 16, rep.Memories)
	assert.Equal(t, []phaseSummary{
		{Phase: phaseNoMemory, tally: tally{Probes: 4, Hit: 2, Ungrounded: 4}},
		{Phase: phaseFirstSeen, tally: tally{Probes: 4, Hit: 4, Right: 4, Top1: 4}},
		{Phase: phaseRecurring, tally: tally{Probes: 7, Hit: 7, Right: 7, Top1: 4}},
		{Phase: phaseStale, tally: tally{Probes: 3, Hit: 3, Right: 3, Top1: 3, Stale: 3}},
	}, rep.Phases)
}

func TestReplayNeedsDeterministicEmbeddings(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "")

	_, err := replay(context.Background(), builtinScenario, replayOptions{Threshold: 0.10, Limit: 5})
	require.ErrorContains(t, err, deterministicEmbeddingsEnv)
}
