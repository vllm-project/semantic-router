//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package candle_binding

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rerankCase mirrors the RAG-style cases in oss/rerank-repro/demo_rerank.py.
// documents[answerIndex] is the doc that truly answers the query; the rest are
// topical / lexical-overlap distractors.
type rerankCase struct {
	query       string
	documents   []string
	answerIndex int
}

var rerankCases = []rerankCase{
	{
		query: "How do I cancel my subscription?",
		documents: []string{
			"To cancel your subscription, open Settings > Billing and click Cancel plan.",
			"Our subscription plans come in monthly and annual billing options.",
			"Orders cannot be cancelled once they have shipped.",
			"Subscriptions renew automatically unless turned off.",
		},
		answerIndex: 0,
	},
	{
		query: "What is the capital of France?",
		documents: []string{
			"Paris is the capital and most populous city of France.",
			"France is a country in Western Europe known for its cuisine.",
			"The French Riviera is a popular travel destination.",
			"Berlin is the capital of Germany.",
		},
		answerIndex: 0,
	},
	{
		query: "How do I get a refund for a cancelled flight?",
		documents: []string{
			"Submit the refund request form within 30 days and the fare returns to your original payment method.",
			"Flights may be cancelled by the airline because of weather or operational issues.",
			"Travelers should arrive at the airport at least two hours before a flight.",
			"Seat upgrades can be purchased at check-in subject to availability.",
		},
		answerIndex: 0,
	},
	{
		query: "What is the return policy for opened electronics?",
		documents: []string{
			"Opened electronics may be returned within 15 days as long as all original accessories are included.",
			"All electronics are covered by a one-year manufacturer warranty against defects.",
			"Most unopened items can be returned within 30 days for a full refund.",
			"Please recycle old electronics at a certified e-waste collection center.",
		},
		answerIndex: 0,
	},
	{
		query: "How many bags can I check for free on an international flight?",
		documents: []string{
			"Economy passengers traveling abroad may check two pieces at no extra cost.",
			"Baggage rules differ between domestic and international flights.",
			"International flights generally begin boarding 45 minutes before departure.",
			"Checked bags must not exceed 23 kg or they incur an overweight fee.",
		},
		answerIndex: 0,
	},
}

// TestCrossEncoderRerankLive exercises the real cross-encoder FFI path end to
// end. It is skipped unless CROSS_ENCODER_MODEL_PATH points at a downloaded
// reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
func TestCrossEncoderRerankLive(t *testing.T) {
	modelPath := os.Getenv("CROSS_ENCODER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set CROSS_ENCODER_MODEL_PATH to a local cross-encoder dir to run this test")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model path %q not found: %v", modelPath, err)
	}

	if err := InitCrossEncoder(modelPath, true); err != nil {
		t.Fatalf("InitCrossEncoder failed: %v", err)
	}
	if !IsCrossEncoderInitialized() {
		t.Fatal("cross-encoder reports not initialized after InitCrossEncoder")
	}

	correct := 0
	rankSum := 0
	for _, c := range rerankCases {
		top1Correct, answerRank := runRerankCase(t, c)
		if top1Correct {
			correct++
		}
		rankSum += answerRank
	}

	n := len(rerankCases)
	t.Logf("SUMMARY: cross-encoder top-1 accuracy %d/%d, avg answer rank %.2f", correct, n, float64(rankSum)/float64(n))

	// The cross-encoder should put the true answer at #1 in the strong majority
	// of cases; we assert a conservative floor so the test is not flaky.
	if correct < n-1 {
		t.Errorf("cross-encoder top-1 accuracy too low: %d/%d", correct, n)
	}
}

// runRerankCase reranks one case, logs the ranking, asserts descending order,
// and reports whether the answer ranked #1 and at what rank it landed.
func runRerankCase(t *testing.T, c rerankCase) (top1Correct bool, answerRank int) {
	t.Helper()
	out, err := RerankDocuments(c.query, c.documents, 0)
	if err != nil {
		t.Fatalf("RerankDocuments(%q) failed: %v", c.query, err)
	}
	if len(out.Matches) != len(c.documents) {
		t.Fatalf("expected %d matches, got %d", len(c.documents), len(out.Matches))
	}

	t.Logf("QUERY: %s", c.query)
	answerRank = -1
	for rank, m := range out.Matches {
		if m.Index == c.answerIndex {
			answerRank = rank + 1
		}
		logRerankMatch(t, rank, m, c)
	}

	assertDescending(t, out.Matches)

	t.Logf("  => top-1 index=%d (answer at rank #%d), %.1fms\n", out.Matches[0].Index, answerRank, out.ProcessingTimeMs)
	return out.Matches[0].Index == c.answerIndex, answerRank
}

// logRerankMatch prints one ranked document line (truncated), marking the answer.
func logRerankMatch(t *testing.T, rank int, m RerankMatch, c rerankCase) {
	t.Helper()
	marker := ""
	if m.Index == c.answerIndex {
		marker = "  <-- correct answer"
	}
	doc := c.documents[m.Index]
	if len(doc) > 60 {
		doc = doc[:60]
	}
	t.Logf("  #%d  score=%.4f  [doc %d] %s%s", rank+1, m.Score, m.Index, doc, marker)
}

// assertDescending fails the test if scores are not sorted in descending order.
func assertDescending(t *testing.T, matches []RerankMatch) {
	t.Helper()
	for i := 1; i < len(matches); i++ {
		if matches[i].Score > matches[i-1].Score {
			t.Errorf("scores not sorted descending at %d: %f > %f", i, matches[i].Score, matches[i-1].Score)
		}
	}
}

// parityGolden is the HuggingFace `transformers` ground truth committed in
// testdata/reranker_parity_golden.json. It is (re)generated by
// tools/reranker-parity/verify_parity.py so this test needs no torch/transformers.
type parityCase struct {
	Query  string    `json:"query"`
	Scores []float64 `json:"scores"`
}

type parityGolden struct {
	Model     string       `json:"model"`
	MaxLength int          `json:"max_length"`
	Cases     []parityCase `json:"cases"`
}

// TestCrossEncoderRerankMatchesTransformersGolden asserts the candle cross-encoder
// produces the same per-document relevance scores as HuggingFace `transformers`
// (within tolerance), using committed golden scores so no Python is needed at test
// time. This is the numeric correctness check behind the "matches transformers"
// claim; it is env-gated like the other live tests.
func TestCrossEncoderRerankMatchesTransformersGolden(t *testing.T) {
	modelPath := os.Getenv("CROSS_ENCODER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set CROSS_ENCODER_MODEL_PATH to a local cross-encoder dir to run this test")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model path %q not found: %v", modelPath, err)
	}
	if err := InitCrossEncoder(modelPath, true); err != nil {
		t.Fatalf("InitCrossEncoder failed: %v", err)
	}

	goldenPath := filepath.Join("testdata", "reranker_parity_golden.json")
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}
	var golden parityGolden
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatalf("parse golden %s: %v", goldenPath, err)
	}
	if len(golden.Cases) != len(rerankCases) {
		t.Fatalf("golden has %d cases, test has %d; regenerate with tools/reranker-parity/verify_parity.py",
			len(golden.Cases), len(rerankCases))
	}

	// candle uses f32 sigmoid; transformers uses f64. Parity is tight in practice,
	// so a small absolute tolerance catches loader/pooler regressions without flaking.
	const tol = 0.02
	for i, c := range rerankCases {
		assertCaseMatchesGolden(t, i, c, golden.Cases[i], tol)
	}
	t.Logf("candle cross-encoder matches transformers golden within %.2f across %d cases", tol, len(rerankCases))
}

// assertCaseMatchesGolden reranks one case with candle and asserts every
// per-document score is within tol of the committed transformers score.
func assertCaseMatchesGolden(t *testing.T, i int, c rerankCase, g parityCase, tol float64) {
	t.Helper()
	if g.Query != c.query {
		t.Fatalf("case %d query mismatch: golden %q vs test %q (out of sync)", i, g.Query, c.query)
	}
	if len(g.Scores) != len(c.documents) {
		t.Fatalf("case %d: golden has %d scores, %d documents", i, len(g.Scores), len(c.documents))
	}

	out, err := RerankDocuments(c.query, c.documents, 0)
	if err != nil {
		t.Fatalf("RerankDocuments(%q) failed: %v", c.query, err)
	}
	got := make([]float64, len(c.documents))
	seen := make([]bool, len(c.documents))
	for _, m := range out.Matches {
		if m.Index < 0 || m.Index >= len(c.documents) {
			t.Fatalf("case %d: match index %d out of range", i, m.Index)
		}
		got[m.Index] = float64(m.Score)
		seen[m.Index] = true
	}
	for idx := range c.documents {
		if !seen[idx] {
			t.Fatalf("case %d: no candle score for doc %d", i, idx)
		}
		if diff := math.Abs(got[idx] - g.Scores[idx]); diff > tol {
			t.Errorf("case %d doc %d: candle %.4f vs transformers %.4f (diff %.4f > tol %.2f)",
				i, idx, got[idx], g.Scores[idx], diff, tol)
		}
	}
}

// TestCrossEncoderRerankLongInputTruncates verifies that very long query/document
// pairs (well beyond BERT's 512 position limit) are truncated by the tokenizer
// instead of overflowing position embeddings and failing at inference (the 500
// failure mode flagged in review). The reranker must return finite scores.
func TestCrossEncoderRerankLongInputTruncates(t *testing.T) {
	modelPath := os.Getenv("CROSS_ENCODER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set CROSS_ENCODER_MODEL_PATH to a local cross-encoder dir to run this test")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model path %q not found: %v", modelPath, err)
	}
	if err := InitCrossEncoder(modelPath, true); err != nil {
		t.Fatalf("InitCrossEncoder failed: %v", err)
	}

	// ~4000 whitespace-separated tokens each: far past max_position_embeddings.
	longQuery := strings.TrimSpace(strings.Repeat("refund cancellation policy ", 1000))
	longDoc := strings.TrimSpace(strings.Repeat("the quick brown fox jumps over ", 1000))
	relevantDoc := "To cancel and get a refund, open Settings > Billing within 30 days."

	out, err := RerankDocuments(longQuery, []string{longDoc, relevantDoc}, 0)
	if err != nil {
		t.Fatalf("RerankDocuments on long inputs failed (expected truncation, not error): %v", err)
	}
	if len(out.Matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(out.Matches))
	}
	for _, m := range out.Matches {
		if math.IsNaN(float64(m.Score)) || math.IsInf(float64(m.Score), 0) {
			t.Fatalf("got non-finite score %v for doc %d", m.Score, m.Index)
		}
	}
	t.Logf("long-input rerank ok: top doc=%d score=%.4f (%.1fms)",
		out.Matches[0].Index, out.Matches[0].Score, out.ProcessingTimeMs)
}
