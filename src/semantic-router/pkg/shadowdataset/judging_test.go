package shadowdataset

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const judgeKey = "judge-key"

// judgedManifest has one example with two shadow arms and the text behind all
// three digests.
func judgedManifest(t *testing.T) (Manifest, map[string]ExampleText) {
	t.Helper()
	manifest := buildOrFail(t, []store.Record{comparedRecord("r1", "ask one", "primary answer",
		shadowOutcome("candidate-a", digestOf("answer a")),
		shadowOutcome("candidate-b", digestOf("answer b")),
	)}, testPolicy())
	example := manifest.Examples[0]
	shadows := make([]string, len(example.Shadows))
	for i, arm := range example.Shadows {
		shadows[i] = map[string]string{"candidate-a": "answer a", "candidate-b": "answer b"}[arm.Model]
	}
	return manifest, map[string]ExampleText{
		example.ID: {Input: "ask one", Primary: "primary answer", Shadows: shadows},
	}
}

func buildTasksOrFail(t *testing.T, m Manifest, texts map[string]ExampleText) JudgeTaskSet {
	t.Helper()
	set, err := BuildJudgeTasks(m, texts, judgeKey)
	if err != nil {
		t.Fatalf("BuildJudgeTasks: %v", err)
	}
	return set
}

func TestBuildJudgeTasksEmitsEveryPairInBothOrders(t *testing.T) {
	manifest, texts := judgedManifest(t)
	set := buildTasksOrFail(t, manifest, texts)

	if set.Counts.Pairs != 2 || set.Counts.Tasks != 4 || len(set.Tasks) != 4 {
		t.Fatalf("counts %+v with %d tasks, want 2 pairs in 4 tasks", set.Counts, len(set.Tasks))
	}
	byPair := map[string][]JudgeTask{}
	for _, task := range set.Tasks {
		byPair[task.Pair] = append(byPair[task.Pair], task)
	}
	for pair, tasks := range byPair {
		if len(tasks) != 2 {
			t.Fatalf("pair %s has %d tasks, want both orders", pair, len(tasks))
		}
		if tasks[0].First != tasks[1].Second || tasks[0].Second != tasks[1].First {
			t.Fatalf("pair %s is not the same two sides swapped: %+v", pair, tasks)
		}
	}
}

// Nothing a judge is handed may name a model or mark the primary. The arm
// labels differ per pair, so the side shared by two pairs of one example, the
// primary, does not repeat.
func TestBuildJudgeTasksCarriesNoModelIdentity(t *testing.T) {
	manifest, texts := judgedManifest(t)
	set := buildTasksOrFail(t, manifest, texts)

	raw, err := json.Marshal(set)
	if err != nil {
		t.Fatalf("marshal tasks: %v", err)
	}
	encoded := string(raw)
	for _, identity := range []string{"primary-model", "candidate-a", "candidate-b", "primary\"", judgeKey} {
		if strings.Contains(encoded, identity) {
			t.Fatalf("judge tasks carry %q: %s", identity, encoded)
		}
	}
	arms := map[string]int{}
	for _, task := range set.Tasks {
		arms[task.First.Arm]++
	}
	if len(arms) != 4 {
		t.Fatalf("saw %d distinct first-slot arms across 4 tasks, want every side labelled per pair", len(arms))
	}
}

// The key alone recovers which side is which, so the mapping never has to be
// published beside the tasks.
func TestArmIDRecoversTheSidesWithTheKey(t *testing.T) {
	manifest, texts := judgedManifest(t)
	set := buildTasksOrFail(t, manifest, texts)
	example := manifest.Examples[0]

	for _, task := range set.Tasks {
		for _, side := range []Candidate{task.First, task.Second} {
			matched := false
			for index := range example.Shadows {
				if side.Arm == ArmID(judgeKey, example.ID, index, true) {
					matched = side.Text == "primary answer"
				}
				if side.Arm == ArmID(judgeKey, example.ID, index, false) {
					matched = side.Text == texts[example.ID].Shadows[index]
				}
			}
			if !matched {
				t.Fatalf("arm %s with text %q maps to no side under the key", side.Arm, side.Text)
			}
		}
	}
}

func TestBuildJudgeTasksIsReproducibleUnderOneKey(t *testing.T) {
	manifest, texts := judgedManifest(t)
	first := buildTasksOrFail(t, manifest, texts)
	if second := buildTasksOrFail(t, manifest, texts); !reflect.DeepEqual(first, second) {
		t.Fatal("the same manifest and key built different tasks")
	}
	other, err := BuildJudgeTasks(manifest, texts, "another-key")
	if err != nil {
		t.Fatalf("BuildJudgeTasks: %v", err)
	}
	if other.Tasks[0].ID == first.Tasks[0].ID {
		t.Fatal("a different key produced the same task identities")
	}
}

// A judge reads text while the manifest holds digests. Text that does not hash
// back to the recorded digest is a truncated excerpt or an answer the router
// rewrote, and judging it would credit the model with something it did not say.
func TestBuildJudgeTasksExcludesTextThatDoesNotMatchItsDigest(t *testing.T) {
	manifest, texts := judgedManifest(t)
	id := manifest.Examples[0].ID
	damaged := texts[id]
	damaged.Shadows = []string{damaged.Shadows[0][:3], damaged.Shadows[1]}
	texts[id] = damaged

	set := buildTasksOrFail(t, manifest, texts)
	if set.Counts.Pairs != 1 || set.Counts.Excluded[ExcludeJudgeTextMismatch] != 1 {
		t.Fatalf("counts %+v, want the truncated pair excluded and the other kept", set.Counts)
	}

	damaged.Primary = "[warning] " + damaged.Primary
	texts[id] = damaged
	set = buildTasksOrFail(t, manifest, texts)
	if set.Counts.Pairs != 0 || set.Counts.Excluded[ExcludeJudgeTextMismatch] != 2 {
		t.Fatalf("counts %+v, want both pairs excluded once the primary was rewritten", set.Counts)
	}
}

func TestBuildJudgeTasksCountsMissingText(t *testing.T) {
	manifest, _ := judgedManifest(t)
	set := buildTasksOrFail(t, manifest, map[string]ExampleText{})
	if set.Counts.Pairs != 0 || set.Counts.Excluded[ExcludeJudgeTextMissing] != 2 {
		t.Fatalf("counts %+v, want both pairs counted as missing text", set.Counts)
	}
}

func TestBuildJudgeTasksFlagsAnAnswerThatNamesItsModel(t *testing.T) {
	manifest := buildOrFail(t, []store.Record{comparedRecord("r1", "ask one", "primary answer",
		shadowOutcome("openai/shadow-candidate", digestOf("I am Shadow-Candidate, happy to help")),
	)}, testPolicy())
	texts := map[string]ExampleText{manifest.Examples[0].ID: {
		Input: "ask one", Primary: "primary answer", Shadows: []string{"I am Shadow-Candidate, happy to help"},
	}}

	for _, task := range buildTasksOrFail(t, manifest, texts).Tasks {
		for _, side := range []Candidate{task.First, task.Second} {
			if want := strings.HasPrefix(side.Text, "I am"); side.NamesOwnModel != want {
				t.Fatalf("side %q flagged %v, want %v", side.Text, side.NamesOwnModel, want)
			}
		}
	}
}

// The manifest seed is published with the manifest, so a blinding key equal to
// it would let any reader recompute the arm mapping.
func TestBuildJudgeTasksRejectsAKeyAnyReaderHolds(t *testing.T) {
	manifest, texts := judgedManifest(t)
	for name, key := range map[string]string{"empty": " ", "manifest seed": manifest.Policy.Seed} {
		t.Run(name, func(t *testing.T) {
			if _, err := BuildJudgeTasks(manifest, texts, key); err == nil {
				t.Fatal("BuildJudgeTasks accepted a key that blinds nothing")
			}
		})
	}
}
