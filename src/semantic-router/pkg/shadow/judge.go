package shadow

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// JudgeOutcome is the normalized result of a blinded comparison
// (issue #3376 M3). Every path has an explicit representation.
type JudgeOutcome string

const (
	// JudgeWinner picks exactly one surviving arm.
	JudgeWinner JudgeOutcome = "winner"
	// JudgeTie means the judge named multiple surviving arms as best.
	JudgeTie JudgeOutcome = "tie"
	// JudgeAbstain means the judge declined to pick (null winner).
	JudgeAbstain JudgeOutcome = "abstain"
	// JudgeMalformed means the judge's response could not be interpreted
	// (unparsable JSON or an unknown arm id).
	JudgeMalformed JudgeOutcome = "malformed"
	// JudgeTimeout means the judge call failed or exceeded its bound; the
	// comparison is abandoned, never the primary route.
	JudgeTimeout JudgeOutcome = "timeout"
	// JudgeInsufficientArms means fewer than two arms completed, so no
	// comparison is meaningful.
	JudgeInsufficientArms JudgeOutcome = "insufficient_arms"
)

// JudgeDecision is the blinded comparison result, ready for Replay evidence.
type JudgeDecision struct {
	Outcome            JudgeOutcome
	WinnerArmID        string
	TieArmIDs          []string
	Reason             string
	JudgeModel         string
	JudgeRubricVersion string
	LatencyMS          int64
}

// Judge performs blinded multi-arm judging. It owns a stable opaque blind-ID
// mapping derived once from the configured arms; judge input therefore never
// exposes model/provider identity.
type Judge struct {
	cfg    config.ShadowJudgeConfig
	armIDs map[string]string
}

// NewJudge builds the stable opaque blind-ID mapping from the configured arms
// (configuration order). IDs stay stable across requests so comparison
// datasets can correlate arms (#3280). The mapping is private to this Judge.
func NewJudge(cfg config.ShadowJudgeConfig, arms []config.ShadowArmConfig) *Judge {
	armIDs := make(map[string]string, len(arms))
	for i, arm := range arms {
		if arm.Name != "" {
			armIDs[arm.Name] = fmt.Sprintf("arm-%d", i+1)
		}
	}
	return &Judge{cfg: cfg, armIDs: armIDs}
}

// BlindID returns the stable opaque identifier for a configured arm. Unknown
// names fall back to themselves only as a defensive default.
func (j *Judge) BlindID(armName string) string {
	if id, ok := j.armIDs[armName]; ok {
		return id
	}
	return armName
}

// JudgeModel and JudgeRubricVersion surface the running judge version for
// Replay provenance.
func (j *Judge) JudgeModel() string         { return j.cfg.Model }
func (j *Judge) JudgeRubricVersion() string { return j.cfg.RubricVersion }

type blindItem struct {
	name    string
	id      string
	content string
}

// Decide runs the blinded judge over the surviving (completed) arms. The
// presentation order is randomized per call so position cannot influence the
// result; tests may inject a fixed order as the optional final argument to
// prove order-independence. Judge failures never propagate to the primary
// path: every failure maps to an explicit JudgeOutcome.
func (j *Judge) Decide(ctx context.Context, question string, results []ArmResult, order ...[]int) JudgeDecision {
	items := j.orderedItems(results, order)

	if len(items) < 2 {
		return JudgeDecision{
			Outcome:            JudgeInsufficientArms,
			JudgeModel:         j.cfg.Model,
			JudgeRubricVersion: j.cfg.RubricVersion,
		}
	}

	start := time.Now()
	body, err := j.buildJudgeBody(question, items)
	if err != nil {
		return JudgeDecision{
			Outcome: JudgeMalformed, Reason: err.Error(),
			JudgeModel: j.cfg.Model, JudgeRubricVersion: j.cfg.RubricVersion,
		}
	}

	decision := j.callJudge(ctx, body)
	decision.JudgeModel = j.cfg.Model
	decision.JudgeRubricVersion = j.cfg.RubricVersion
	decision.LatencyMS = time.Since(start).Milliseconds()
	return decision
}

// orderedItems narrows results to completed arms and applies the call's
// presentation order (random by default, injectable for tests). The order is
// the only thing that varies between repeats: the blind ids and the judge
// response mapping stay stable.
func (j *Judge) orderedItems(results []ArmResult, order [][]int) []blindItem {
	completed := make([]ArmResult, 0, len(results))
	for _, res := range results {
		if res.Outcome == OutcomeCompleted {
			completed = append(completed, res)
		}
	}
	items := make([]blindItem, len(completed))
	idx := make([]int, len(completed))
	for i, res := range completed {
		items[i] = blindItem{name: res.Arm, id: j.BlindID(res.Arm), content: res.Content}
		idx[i] = i
	}
	if len(order) > 0 && len(order[0]) == len(items) {
		idx = order[0]
	} else if len(items) > 1 {
		rand.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
	}
	ordered := make([]blindItem, len(items))
	for i, p := range idx {
		ordered[i] = items[p]
	}
	return ordered
}

func (j *Judge) buildJudgeBody(question string, items []blindItem) ([]byte, error) {
	var sb strings.Builder
	sb.WriteString("Question: ")
	if question == "" {
		sb.WriteString("(no question text available)")
	} else {
		sb.WriteString(truncate([]byte(question)))
	}
	for _, it := range items {
		fmt.Fprintf(&sb, "\n\n[%s]\n%s", it.id, truncate([]byte(it.content)))
	}
	system := "You compare candidate answers by quality, not by any identity or position. " +
		"Candidate ids are opaque: ignore their order, treat every id equally, and judge only the answer text."
	payload := map[string]interface{}{
		"model":       j.cfg.Model,
		"temperature": 0,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": sb.String() +
				"\n\nReply with JSON only: {\"winner\":\"<id>\"} for a single best answer; " +
				"{\"tie\":[\"<id>\",\"<id>\"]} when several tie for best; " +
				"{\"winner\":null} when none is clearly best."},
		},
	}
	return json.Marshal(payload)
}

// callJudge performs the judge chat call and maps every failure mode to an
// explicit judge outcome.
func (j *Judge) callJudge(ctx context.Context, body []byte) JudgeDecision {
	judgeCtx := ctx
	cancel := func() {}
	if j.cfg.TimeoutSeconds > 0 {
		judgeCtx, cancel = context.WithTimeout(ctx, time.Duration(j.cfg.TimeoutSeconds)*time.Second)
	}
	defer cancel()

	req, err := http.NewRequestWithContext(judgeCtx, http.MethodPost, j.cfg.Endpoint, bytes.NewReader(body))
	if err != nil {
		return JudgeDecision{Outcome: JudgeTimeout, Reason: "new request: " + err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		if judgeCtx.Err() != nil {
			return JudgeDecision{Outcome: JudgeTimeout, Reason: judgeCtx.Err().Error()}
		}
		return JudgeDecision{Outcome: JudgeTimeout, Reason: "judge request failed: " + err.Error()}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil || resp.StatusCode != http.StatusOK {
		return JudgeDecision{Outcome: JudgeTimeout, Reason: fmt.Sprintf("judge call failed status=%d", resp.StatusCode)}
	}
	return j.parseJudgeResponse(raw)
}

// parseJudgeResponse maps a judge body into a decision: single winner ->
// mapping, tie list -> tie, null -> abstain, unparsable or unknown id ->
// malformed.
func (j *Judge) parseJudgeResponse(raw []byte) JudgeDecision {
	text := strings.TrimSpace(extractContent(raw))
	parsed := struct {
		Winner *string  `json:"winner"`
		Tie    []string `json:"tie"`
	}{}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return JudgeDecision{Outcome: JudgeMalformed, Reason: "judge response not JSON: " + err.Error()}
	}
	if len(parsed.Tie) > 0 {
		for _, id := range parsed.Tie {
			if !j.knownBlindID(strings.TrimSpace(id)) {
				return JudgeDecision{Outcome: JudgeMalformed, Reason: "unknown tie id: " + id}
			}
		}
		return JudgeDecision{Outcome: JudgeTie, TieArmIDs: parsed.Tie}
	}
	if parsed.Winner == nil {
		return JudgeDecision{Outcome: JudgeAbstain, Reason: "judge abstained"}
	}
	winner := strings.TrimSpace(*parsed.Winner)
	if !j.knownBlindID(winner) {
		return JudgeDecision{Outcome: JudgeMalformed, Reason: "unknown winner id: " + winner}
	}
	return JudgeDecision{Outcome: JudgeWinner, WinnerArmID: winner}
}

func (j *Judge) knownBlindID(id string) bool {
	for _, candidate := range j.armIDs {
		if candidate == id {
			return true
		}
	}
	return false
}

// extractContent pulls the assistant text out of an OpenAI-style chat response.
func extractContent(raw []byte) string {
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return string(raw)
	}
	if len(completion.Choices) == 0 {
		return string(raw)
	}
	return completion.Choices[0].Message.Content
}
