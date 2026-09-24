package shadowdataset

import (
	"fmt"
	"sort"
	"strings"
)

// JudgeTasksVersion identifies the judge task shape. A change to what
// BuildJudgeTasks emits or to how it derives opaque identities changes this
// string, because judgments collected under one shape cannot be read under
// another.
const JudgeTasksVersion = "shadow-judge-tasks.v1"

// Reasons a pair is left out of the judge tasks. A judge reads text while a
// manifest holds digests, so a pair is judged only when both texts hash back to
// the digests the manifest recorded. Anything else would credit a model with a
// fragment of its answer, or with text the router wrote over it.
const (
	ExcludeJudgeTextMissing  = "arm_text_missing"
	ExcludeJudgeTextMismatch = "arm_text_digest_mismatch"
)

// ExampleText is the text behind one example's digests. Shadows follows the
// order of Example.Shadows.
type ExampleText struct {
	Input   string
	Primary string
	Shadows []string
}

// Candidate is one side of a judge task. Arm is opaque: it names neither the
// model nor whether the side is the primary, and it is different in every pair,
// so a judge cannot follow a model, or the primary, from one pair to the next.
type Candidate struct {
	Arm  string `json:"arm"`
	Text string `json:"text"`
	// NamesOwnModel is set when the text names the model that wrote it. Opaque
	// labels do not blind a judge to an answer that introduces itself, so the
	// leak is recorded for the consumer to exclude or report.
	NamesOwnModel bool `json:"names_own_model,omitempty"`
}

// JudgeTask asks a judge to compare one primary answer with one shadow answer.
// Every pair is emitted twice with the sides swapped, under one Pair ID, so a
// verdict that follows the slot rather than the answer is measurable.
type JudgeTask struct {
	ID     string    `json:"id"`
	Pair   string    `json:"pair"`
	Input  string    `json:"input"`
	First  Candidate `json:"first"`
	Second Candidate `json:"second"`
}

// JudgeTaskCounts reports what BuildJudgeTasks kept and dropped, by pair.
type JudgeTaskCounts struct {
	Pairs    int            `json:"pairs"`
	Tasks    int            `json:"tasks"`
	Excluded map[string]int `json:"excluded,omitempty"`
}

// JudgeTaskSet is what a judge is handed. It carries the manifest digest for
// lineage and nothing that maps an opaque arm back to a model.
type JudgeTaskSet struct {
	Version        string          `json:"version"`
	ManifestDigest string          `json:"manifest_digest"`
	Counts         JudgeTaskCounts `json:"counts"`
	Tasks          []JudgeTask     `json:"tasks"`
}

// BuildJudgeTasks turns a manifest and the text behind it into blinded pairwise
// tasks. The key derives every opaque identity. It is never written into the
// set, so the mapping from arm to model stays with whoever holds the key, and
// ArmID recomputes it when judgments come back.
func BuildJudgeTasks(m Manifest, texts map[string]ExampleText, key string) (JudgeTaskSet, error) {
	if strings.TrimSpace(key) == "" {
		return JudgeTaskSet{}, fmt.Errorf("a blinding key is required")
	}
	if key == m.Policy.Seed {
		return JudgeTaskSet{}, fmt.Errorf("the blinding key must differ from the manifest seed, which is published")
	}

	set := JudgeTaskSet{
		Version:        JudgeTasksVersion,
		ManifestDigest: m.Digest,
		Counts:         JudgeTaskCounts{Excluded: map[string]int{}},
	}
	for _, example := range m.Examples {
		text, found := texts[example.ID]
		for index, shadow := range example.Shadows {
			reason := pairTextFault(example, text, found, index)
			if reason != "" {
				set.Counts.Excluded[reason]++
				continue
			}
			primary := Candidate{
				Arm:           ArmID(key, example.ID, index, true),
				Text:          text.Primary,
				NamesOwnModel: namesModel(text.Primary, example.Primary.Model),
			}
			candidate := Candidate{
				Arm:           ArmID(key, example.ID, index, false),
				Text:          text.Shadows[index],
				NamesOwnModel: namesModel(text.Shadows[index], shadow.Model),
			}
			pair := opaqueID(key, "pair", example.ID, fmt.Sprint(index))
			set.Tasks = append(set.Tasks,
				JudgeTask{ID: opaqueID(key, "task", pair, "0"), Pair: pair, Input: text.Input, First: primary, Second: candidate},
				JudgeTask{ID: opaqueID(key, "task", pair, "1"), Pair: pair, Input: text.Input, First: candidate, Second: primary},
			)
			set.Counts.Pairs++
		}
	}

	// Ordering by opaque ID rather than by example keeps the two orders of one
	// pair, and the primary side, from sitting in a predictable position.
	sort.Slice(set.Tasks, func(i, j int) bool { return set.Tasks[i].ID < set.Tasks[j].ID })
	set.Counts.Tasks = len(set.Tasks)
	if len(set.Counts.Excluded) == 0 {
		set.Counts.Excluded = nil
	}
	return set, nil
}

// ArmID is the opaque label of one side of the pair that compares the primary
// of an example with the shadow at that index of Example.Shadows.
func ArmID(key, exampleID string, shadow int, primary bool) string {
	return opaqueID(key, "arm", exampleID, fmt.Sprint(shadow), fmt.Sprint(primary))
}

func pairTextFault(example Example, text ExampleText, found bool, shadow int) string {
	if !found || text.Primary == "" || shadow >= len(text.Shadows) || text.Shadows[shadow] == "" {
		return ExcludeJudgeTextMissing
	}
	if digestOf(text.Primary) != example.Primary.OutputDigest ||
		digestOf(text.Shadows[shadow]) != example.Shadows[shadow].OutputDigest {
		return ExcludeJudgeTextMismatch
	}
	return ""
}

// namesModel reports whether text names a model, by its full reference or by
// the name after its provider prefix, ignoring case.
func namesModel(text, model string) bool {
	lowered := strings.ToLower(text)
	name := strings.ToLower(model)
	if name == "" {
		return false
	}
	if strings.Contains(lowered, name) {
		return true
	}
	_, short, found := strings.Cut(name, "/")
	return found && short != "" && strings.Contains(lowered, short)
}

func opaqueID(key string, parts ...string) string {
	return digestOf(key + "\x00" + strings.Join(parts, "\x00"))[:16]
}
