package shadowdataset

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// JudgmentsVersion identifies the judgment record shape.
const JudgmentsVersion = "shadow-judgments.v1"

// Judgment outcomes. A judge that picks a side names the arm, not the slot, so
// the two orders of one pair compare directly. Anything short of a pick is kept
// as an outcome of its own rather than folded into a tie or dropped, because an
// abstention or a malformed answer is evidence about the judge.
const (
	JudgmentPreferred = "preferred"
	JudgmentTie       = "tie"
	JudgmentAbstain   = "abstain"
	JudgmentMalformed = "malformed"
)

// JudgeIdentity names the judge and the rubric it scored under. Scores from two
// judges, or from one judge under two rubrics, are different measurements.
type JudgeIdentity struct {
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
	RubricVersion string `json:"rubric_version"`
}

// Judgment is one verdict on one task. It has no field for the judge's
// reasoning, and a record that carries one is refused when decoded, so hidden
// reasoning cannot ride along into a published judgment set.
type Judgment struct {
	Task      string `json:"task"`
	Outcome   string `json:"outcome"`
	Preferred string `json:"preferred,omitempty"`
	// Score and Confidence are bounded to [0,1] when present. Neither is a
	// label: a low-confidence pick stays a low-confidence pick.
	Score           *float64 `json:"score,omitempty"`
	Confidence      *float64 `json:"confidence,omitempty"`
	MalformedReason string   `json:"malformed_reason,omitempty"`
	InputTokens     int64    `json:"input_tokens,omitempty"`
	OutputTokens    int64    `json:"output_tokens,omitempty"`
	CostUSD         float64  `json:"cost_usd,omitempty"`
}

// JudgmentSet is what comes back from a judge run over one task set.
type JudgmentSet struct {
	Version        string        `json:"version"`
	ManifestDigest string        `json:"manifest_digest"`
	Judge          JudgeIdentity `json:"judge"`
	Judgments      []Judgment    `json:"judgments"`
}

// DecodeJudgments reads a judgment set and refuses any field the record does
// not define, which is how a reasoning field is kept out.
func DecodeJudgments(r io.Reader) (JudgmentSet, error) {
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()
	var set JudgmentSet
	if err := decoder.Decode(&set); err != nil {
		return JudgmentSet{}, fmt.Errorf("decode judgments: %w", err)
	}
	return set, nil
}

// ValidateJudgments rejects a judgment set that cannot be read against the
// tasks it claims to answer. It runs before anything is published, so a set
// with a verdict for a task nobody was asked, or a pick of an arm the task did
// not show, never becomes evidence.
func ValidateJudgments(tasks JudgeTaskSet, set JudgmentSet) error {
	if set.Version != JudgmentsVersion {
		return fmt.Errorf("judgments version %q, want %q", set.Version, JudgmentsVersion)
	}
	if set.ManifestDigest != tasks.ManifestDigest {
		return fmt.Errorf("judgments answer manifest %q, tasks came from %q", set.ManifestDigest, tasks.ManifestDigest)
	}
	if strings.TrimSpace(set.Judge.Model) == "" || strings.TrimSpace(set.Judge.ModelRevision) == "" ||
		strings.TrimSpace(set.Judge.RubricVersion) == "" {
		return fmt.Errorf("judgments must name the judge model, its revision and the rubric version")
	}

	byID := make(map[string]JudgeTask, len(tasks.Tasks))
	for _, task := range tasks.Tasks {
		byID[task.ID] = task
	}
	seen := make(map[string]bool, len(set.Judgments))
	for _, judgment := range set.Judgments {
		task, found := byID[judgment.Task]
		if !found {
			return fmt.Errorf("judgment for unknown task %q", judgment.Task)
		}
		if seen[judgment.Task] {
			return fmt.Errorf("task %q is judged more than once", judgment.Task)
		}
		seen[judgment.Task] = true
		if err := judgment.validate(task); err != nil {
			return fmt.Errorf("task %q: %w", judgment.Task, err)
		}
	}
	return nil
}

func (j Judgment) validate(task JudgeTask) error {
	switch j.Outcome {
	case JudgmentPreferred:
		if j.Preferred != task.First.Arm && j.Preferred != task.Second.Arm {
			return fmt.Errorf("preferred arm %q is not one the task showed", j.Preferred)
		}
	case JudgmentTie, JudgmentAbstain, JudgmentMalformed:
		if j.Preferred != "" {
			return fmt.Errorf("outcome %s names a preferred arm", j.Outcome)
		}
	default:
		return fmt.Errorf("unknown outcome %q", j.Outcome)
	}
	if (j.Outcome == JudgmentMalformed) != (j.MalformedReason != "") {
		return fmt.Errorf("a malformed reason goes with a malformed outcome and nothing else")
	}
	if outsideUnit(j.Score) || outsideUnit(j.Confidence) {
		return fmt.Errorf("score and confidence must lie in [0,1]")
	}
	if j.InputTokens < 0 || j.OutputTokens < 0 || j.CostUSD < 0 {
		return fmt.Errorf("budget use cannot be negative")
	}
	return nil
}

func outsideUnit(value *float64) bool {
	return value != nil && (*value < 0 || *value > 1)
}

// JudgeReport summarizes a validated judgment set without the key, so it names
// no model and can be published beside the tasks.
type JudgeReport struct {
	Outcomes map[string]int `json:"outcomes"`
	// FirstSlotPicks and SecondSlotPicks count preferred verdicts by where the
	// picked arm sat. Every pair is shown in both orders, so an unbiased judge
	// splits them evenly.
	FirstSlotPicks  int `json:"first_slot_picks"`
	SecondSlotPicks int `json:"second_slot_picks"`
	// A pair judged in both orders is consistent when both verdicts pick the
	// same arm or both tie, and slot-following when each order picks whichever
	// arm sat in the same slot. Every other combination is inconsistent.
	ConsistentPairs     int     `json:"consistent_pairs"`
	SlotFollowingPairs  int     `json:"slot_following_pairs"`
	InconsistentPairs   int     `json:"inconsistent_pairs"`
	IncompletePairs     int     `json:"incomplete_pairs"`
	IdentityLeakedTasks int     `json:"identity_leaked_tasks"`
	InputTokens         int64   `json:"input_tokens"`
	OutputTokens        int64   `json:"output_tokens"`
	CostUSD             float64 `json:"cost_usd"`
}

// ReportJudgments measures position bias and identity leakage over a set that
// ValidateJudgments accepted.
func ReportJudgments(tasks JudgeTaskSet, set JudgmentSet) JudgeReport {
	report := JudgeReport{Outcomes: map[string]int{}}
	byTask := make(map[string]Judgment, len(set.Judgments))
	for _, judgment := range set.Judgments {
		byTask[judgment.Task] = judgment
		report.Outcomes[judgment.Outcome]++
		report.InputTokens += judgment.InputTokens
		report.OutputTokens += judgment.OutputTokens
		report.CostUSD += judgment.CostUSD
	}

	pairs := map[string][]JudgeTask{}
	for _, task := range tasks.Tasks {
		pairs[task.Pair] = append(pairs[task.Pair], task)
		judgment, judged := byTask[task.ID]
		if !judged {
			continue
		}
		if task.First.NamesOwnModel || task.Second.NamesOwnModel {
			report.IdentityLeakedTasks++
		}
		if judgment.Outcome == JudgmentPreferred {
			if judgment.Preferred == task.First.Arm {
				report.FirstSlotPicks++
			} else {
				report.SecondSlotPicks++
			}
		}
	}
	for _, orders := range pairs {
		report.countPair(orders, byTask)
	}
	return report
}

func (r *JudgeReport) countPair(orders []JudgeTask, byTask map[string]Judgment) {
	if len(orders) != 2 {
		r.IncompletePairs++
		return
	}
	a, aJudged := byTask[orders[0].ID]
	b, bJudged := byTask[orders[1].ID]
	decisive := func(j Judgment) bool { return j.Outcome == JudgmentPreferred || j.Outcome == JudgmentTie }
	if !aJudged || !bJudged || !decisive(a) || !decisive(b) {
		r.IncompletePairs++
		return
	}
	switch {
	case a.Outcome == JudgmentTie && b.Outcome == JudgmentTie, a.Preferred != "" && a.Preferred == b.Preferred:
		r.ConsistentPairs++
	case a.Preferred == orders[0].First.Arm && b.Preferred == orders[1].First.Arm,
		a.Preferred == orders[0].Second.Arm && b.Preferred == orders[1].Second.Arm:
		r.SlotFollowingPairs++
	default:
		r.InconsistentPairs++
	}
}
