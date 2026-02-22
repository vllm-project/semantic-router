package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Data file schema -- mirrors testdata/evaluation_dataset.json
// ---------------------------------------------------------------------------

type evalDatasetFile struct {
	Description string         `json:"description"`
	Config      evalConfig     `json:"config"`
	Entries     []evalEntry    `json:"entries"`
	Queries     []evalQueryDef `json:"queries"`
}

type evalConfig struct {
	K              int       `json:"k"`
	Threshold      float32   `json:"threshold"`
	Alphas         []float32 `json:"alphas"`
	MaxDepth       int       `json:"max_depth"`
	ScorePropAlpha float32   `json:"score_prop_alpha"`
}

type evalEntry struct {
	ID      string `json:"id"`
	Cluster string `json:"cluster"`
	Content string `json:"content"`
}

type evalQueryDef struct {
	Query         string `json:"query"`
	TargetCluster string `json:"target_cluster"`
}

// loadEvalDataset reads the JSON dataset from testdata/.
// Override the file by setting EVAL_DATASET_PATH env var.
func loadEvalDataset(t *testing.T) *evalDatasetFile {
	t.Helper()

	path := os.Getenv("EVAL_DATASET_PATH")
	if path == "" {
		_, thisFile, _, _ := runtime.Caller(0)
		path = filepath.Join(filepath.Dir(thisFile), "testdata", "evaluation_dataset.json")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read evaluation dataset from %s: %v", path, err)
	}

	var ds evalDatasetFile
	if err := json.Unmarshal(data, &ds); err != nil {
		t.Fatalf("failed to parse evaluation dataset: %v", err)
	}

	if len(ds.Entries) == 0 {
		t.Fatal("evaluation dataset has no entries")
	}
	if len(ds.Queries) == 0 {
		t.Fatal("evaluation dataset has no queries")
	}

	if ds.Config.K <= 0 {
		ds.Config.K = 5
	}
	if ds.Config.Threshold <= 0 {
		ds.Config.Threshold = 0.3
	}
	if ds.Config.MaxDepth <= 0 {
		ds.Config.MaxDepth = 3
	}
	if ds.Config.ScorePropAlpha <= 0 {
		ds.Config.ScorePropAlpha = 0.6
	}
	if len(ds.Config.Alphas) == 0 {
		ds.Config.Alphas = []float32{1.0, 0.7, 0.5, 0.3}
	}

	return &ds
}

// buildStores creates a flat store and an enriched hierarchical store from the dataset.
func buildStores(t *testing.T, ds *evalDatasetFile) (flat, hier *InMemoryStore) {
	t.Helper()
	ctx := context.Background()
	now := time.Now()

	flat = newTestInMemoryStore()
	hier = newTestInMemoryStore()

	for _, e := range ds.Entries {
		memF := &Memory{ID: e.ID, Type: MemoryTypeSemantic, Content: e.Content, UserID: "user1", CreatedAt: now}
		requireNoErr(t, flat.Store(ctx, memF))

		memH := &Memory{ID: e.ID, Type: MemoryTypeSemantic, Content: e.Content, UserID: "user1", CreatedAt: now}
		requireNoErr(t, hier.Store(ctx, memH))
		requireNoErr(t, EnrichMemoryBeforeStore(ctx, hier, memH, hier.embeddingConfig, CategorizerConfig{}))
		requireNoErr(t, hier.Update(ctx, memH.ID, memH))
	}
	return
}

// buildLabelIndex returns cluster->id mapping and per-cluster size counts.
func buildLabelIndex(ds *evalDatasetFile) (labelMap map[string]string, clusterSizes map[string]int) {
	labelMap = make(map[string]string, len(ds.Entries))
	clusterSizes = make(map[string]int)
	for _, e := range ds.Entries {
		labelMap[e.ID] = e.Cluster
		clusterSizes[e.Cluster]++
	}
	return
}

// ---------------------------------------------------------------------------
// Metrics helpers
// ---------------------------------------------------------------------------

func precisionAtK(results []*RetrieveResult, k int, target string, labels map[string]string) float64 {
	n := k
	if n > len(results) {
		n = len(results)
	}
	if n == 0 {
		return 0
	}
	hits := 0
	for _, r := range results[:n] {
		if labels[r.Memory.ID] == target {
			hits++
		}
	}
	return float64(hits) / float64(n)
}

func recallAtK(results []*RetrieveResult, k int, target string, labels map[string]string, clusterSize int) float64 {
	n := k
	if n > len(results) {
		n = len(results)
	}
	if n == 0 || clusterSize == 0 {
		return 0
	}
	hits := 0
	for _, r := range results[:n] {
		if labels[r.Memory.ID] == target {
			hits++
		}
	}
	return float64(hits) / float64(clusterSize)
}

func clusterPurity(results []*RetrieveResult, k int, labels map[string]string) float64 {
	n := k
	if n > len(results) {
		n = len(results)
	}
	if n == 0 {
		return 0
	}
	topCluster := labels[results[0].Memory.ID]
	if topCluster == "" {
		return 0
	}
	same := 0
	for _, r := range results[:n] {
		if labels[r.Memory.ID] == topCluster {
			same++
		}
	}
	return float64(same) / float64(n)
}

func injectedTokenCount(results []*RetrieveResult) int {
	total := 0
	for _, r := range results {
		if r.Memory != nil && r.Memory.Content != "" {
			total += len(strings.Fields(r.Memory.Content))
		}
		for _, rel := range r.Related {
			if rel.Abstract != "" {
				total += len(strings.Fields(rel.Abstract))
			}
		}
	}
	return total
}

func injectedTokenCountAbstractOnly(results []*RetrieveResult) int {
	total := 0
	for _, r := range results {
		if r.Memory != nil {
			text := r.Memory.Abstract
			if text == "" {
				text = r.Memory.Content
			}
			total += len(strings.Fields(text))
		}
		for _, rel := range r.Related {
			if rel.Abstract != "" {
				total += len(strings.Fields(rel.Abstract))
			}
		}
	}
	return total
}

func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max]
}

func requireNoErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test: Precision / Recall comparison
// ---------------------------------------------------------------------------

func TestComparison_PrecisionRecall(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, sizes := buildLabelIndex(ds)
	flatStore, hierStore := buildStores(t, ds)

	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	catCount := 0
	hierStore.mu.RLock()
	for _, m := range hierStore.memories {
		if m.IsCategory {
			catCount++
		}
	}
	hierStore.mu.RUnlock()

	t.Logf("Dataset: %d entries, %d clusters, %d category nodes", len(ds.Entries), len(sizes), catCount)
	t.Logf("Config: k=%d threshold=%.2f maxDepth=%d alpha=%.2f", k, thr, ds.Config.MaxDepth, ds.Config.ScorePropAlpha)
	t.Log("")

	t.Logf("%-80s  %-12s  %7s %7s  %7s %7s  %5s %5s",
		"Query", "Cluster", "Flat-P", "Hier-P", "Flat-R", "Hier-R", "F-hit", "H-hit")

	var sumFP, sumFR, sumHP, sumHR float64

	for _, q := range ds.Queries {
		flatOpts := RetrieveOptions{Query: q.Query, UserID: "user1", Limit: k, Threshold: thr}

		flatRes, err := flatStore.Retrieve(ctx, flatOpts)
		requireNoErr(t, err)
		hierRes, err := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts,
			MaxDepth:        ds.Config.MaxDepth,
			ScorePropAlpha:  ds.Config.ScorePropAlpha,
		})
		requireNoErr(t, err)

		fp := precisionAtK(flatRes, k, q.TargetCluster, labels)
		fr := recallAtK(flatRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		hp := precisionAtK(hierRes, k, q.TargetCluster, labels)
		hr := recallAtK(hierRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])

		fHits, hHits := 0, 0
		for _, r := range flatRes {
			if labels[r.Memory.ID] == q.TargetCluster {
				fHits++
			}
		}
		for _, r := range hierRes {
			if labels[r.Memory.ID] == q.TargetCluster {
				hHits++
			}
		}

		t.Logf("%-80s  %-12s  %7.2f %7.2f  %7.2f %7.2f  %5d %5d",
			truncateStr(q.Query, 78), q.TargetCluster, fp, hp, fr, hr, fHits, hHits)

		sumFP += fp
		sumFR += fr
		sumHP += hp
		sumHR += hr
	}

	nq := float64(len(ds.Queries))
	avgFP, avgFR := sumFP/nq, sumFR/nq
	avgHP, avgHR := sumHP/nq, sumHR/nq

	t.Log("")
	t.Logf("Averages over %d queries (k=%d):", len(ds.Queries), k)
	t.Logf("  Flat  — P@%d: %.4f  R@%d: %.4f", k, avgFP, k, avgFR)
	t.Logf("  Hier  — P@%d: %.4f  R@%d: %.4f", k, avgHP, k, avgHR)
	t.Log("")

	logDelta(t, "Precision", avgFP, avgHP)
	logDelta(t, "Recall", avgFR, avgHR)
}

// ---------------------------------------------------------------------------
// Test: Token reduction via L0 abstracts
// ---------------------------------------------------------------------------

func TestComparison_TokenReduction(t *testing.T) {
	ds := loadEvalDataset(t)
	ctx := context.Background()
	now := time.Now()
	k := ds.Config.K
	thr := ds.Config.Threshold

	store := newTestInMemoryStore()
	for _, e := range ds.Entries {
		mem := &Memory{ID: e.ID, Type: MemoryTypeSemantic, Content: e.Content, UserID: "user1", CreatedAt: now}
		requireNoErr(t, store.Store(ctx, mem))
		requireNoErr(t, EnrichMemoryBeforeStore(ctx, store, mem, store.embeddingConfig, CategorizerConfig{}))
		requireNoErr(t, store.Update(ctx, mem.ID, mem))
	}

	t.Logf("%-80s  %10s %10s %8s", "Query", "FullTokens", "L0Tokens", "Savings")

	var totalFull, totalL0 int

	for _, q := range ds.Queries {
		results, err := store.Retrieve(ctx, RetrieveOptions{
			Query: q.Query, UserID: "user1", Limit: k, Threshold: thr,
		})
		requireNoErr(t, err)

		full := injectedTokenCount(results)
		l0 := injectedTokenCountAbstractOnly(results)
		totalFull += full
		totalL0 += l0

		pct := float64(0)
		if full > 0 {
			pct = float64(full-l0) / float64(full) * 100
		}
		t.Logf("%-80s  %10d %10d %7.1f%%", truncateStr(q.Query, 78), full, l0, pct)
	}

	t.Log("")
	overallPct := float64(0)
	if totalFull > 0 {
		overallPct = float64(totalFull-totalL0) / float64(totalFull) * 100
	}
	t.Logf("Total: full=%d tokens, L0=%d tokens, savings=%.1f%%", totalFull, totalL0, overallPct)
	t.Log("")

	switch {
	case totalL0 < totalFull:
		t.Logf("RESULT: L0 abstracts reduce injected tokens by %.1f%%", overallPct)
	case totalL0 == totalFull:
		t.Log("RESULT: No token savings — content is short enough that abstracts equal full content")
	default:
		t.Log("RESULT: L0 abstracts are LONGER than full content (unexpected)")
	}
}

// ---------------------------------------------------------------------------
// Test: Score propagation alpha sweep
// ---------------------------------------------------------------------------

func TestComparison_ScorePropagation(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, _ := buildLabelIndex(ds)
	ctx := context.Background()
	now := time.Now()
	k := ds.Config.K
	thr := ds.Config.Threshold
	alphas := ds.Config.Alphas

	store := newTestInMemoryStore()
	for _, e := range ds.Entries {
		mem := &Memory{ID: e.ID, Type: MemoryTypeSemantic, Content: e.Content, UserID: "user1", CreatedAt: now}
		requireNoErr(t, store.Store(ctx, mem))
		requireNoErr(t, EnrichMemoryBeforeStore(ctx, store, mem, store.embeddingConfig, CategorizerConfig{}))
		requireNoErr(t, store.Update(ctx, mem.ID, mem))
	}

	t.Log("alpha=1.0 → pure child score (no propagation)")
	t.Log("alpha<1.0 → blends parent category score into child score")
	t.Log("")

	header := fmt.Sprintf("%-80s  %-12s", "Query", "Cluster")
	for _, a := range alphas {
		header += fmt.Sprintf("  a=%.1f", a)
	}
	t.Log(header)

	avgByAlpha := make([]float64, len(alphas))

	for _, q := range ds.Queries {
		line := fmt.Sprintf("%-80s  %-12s", truncateStr(q.Query, 78), q.TargetCluster)
		for ai, alpha := range alphas {
			results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
				RetrieveOptions: RetrieveOptions{
					Query: q.Query, UserID: "user1", Limit: k, Threshold: thr,
				},
				MaxDepth:       ds.Config.MaxDepth,
				ScorePropAlpha: alpha,
			})
			requireNoErr(t, err)

			p := precisionAtK(results, k, q.TargetCluster, labels)
			avgByAlpha[ai] += p
			line += fmt.Sprintf("  %5.2f", p)
		}
		t.Log(line)
	}

	t.Log("")
	nq := float64(len(ds.Queries))
	bestAlpha, bestAvg := alphas[0], avgByAlpha[0]
	for ai, a := range alphas {
		avg := avgByAlpha[ai] / nq
		t.Logf("  alpha=%.1f → avg P@%d = %.4f", a, k, avg)
		if avgByAlpha[ai] > bestAvg {
			bestAvg = avgByAlpha[ai]
			bestAlpha = a
		}
	}
	t.Log("")

	if bestAlpha < 1.0 {
		t.Logf("RESULT: Score propagation (alpha=%.1f) improved precision over no-propagation (alpha=1.0)", bestAlpha)
	} else {
		t.Log("RESULT: Score propagation did NOT improve precision — pure child scores (alpha=1.0) are best or tied")
	}
}

// ---------------------------------------------------------------------------
// Test: Cluster coherence (purity)
// ---------------------------------------------------------------------------

func TestComparison_ClusterCoherence(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, _ := buildLabelIndex(ds)
	flatStore, hierStore := buildStores(t, ds)

	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	t.Logf("%-80s  %8s %8s", "Query", "FlatPure", "HierPure")

	var sumFP, sumHP float64

	for _, q := range ds.Queries {
		flatRes, _ := flatStore.Retrieve(ctx, RetrieveOptions{
			Query: q.Query, UserID: "user1", Limit: k, Threshold: thr,
		})
		hierRes, _ := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: RetrieveOptions{
				Query: q.Query, UserID: "user1", Limit: k, Threshold: thr,
			},
			MaxDepth:       ds.Config.MaxDepth,
			ScorePropAlpha: ds.Config.ScorePropAlpha,
		})

		fp := clusterPurity(flatRes, k, labels)
		hp := clusterPurity(hierRes, k, labels)
		sumFP += fp
		sumHP += hp
		t.Logf("%-80s  %8.2f %8.2f", truncateStr(q.Query, 78), fp, hp)
	}

	nq := float64(len(ds.Queries))
	t.Log("")
	t.Logf("Average cluster purity: Flat=%.4f  Hier=%.4f", sumFP/nq, sumHP/nq)
	t.Log("")
	logDelta(t, "Cluster purity", sumFP/nq, sumHP/nq)
}

// ---------------------------------------------------------------------------
// Token-count unit test (does not depend on data file)
// ---------------------------------------------------------------------------

func TestTokenCountHelpers(t *testing.T) {
	results := []*RetrieveResult{
		{Memory: &Memory{Content: "one two three", Abstract: "one"}},
		{Memory: &Memory{Content: "four five six seven", Abstract: "four five"}},
	}
	full := injectedTokenCount(results)
	l0 := injectedTokenCountAbstractOnly(results)

	if full != 7 {
		t.Errorf("expected 7 full tokens, got %d", full)
	}
	if l0 != 3 {
		t.Errorf("expected 3 L0 tokens, got %d", l0)
	}
	if l0 > full {
		t.Error("L0 tokens should not exceed full tokens when abstracts are shorter")
	}
}

// ---------------------------------------------------------------------------
// Reporting helper
// ---------------------------------------------------------------------------

func logDelta(t *testing.T, metric string, flat, hier float64) {
	t.Helper()
	switch {
	case hier > flat:
		delta := hier - flat
		pct := float64(0)
		if flat > 0 {
			pct = delta / flat * 100
		}
		t.Logf("RESULT: Hierarchical %s is HIGHER by %.4f (%.1f%%)", metric, delta, pct)
	case hier < flat:
		delta := flat - hier
		pct := float64(0)
		if flat > 0 {
			pct = delta / flat * 100
		}
		t.Logf("RESULT: Hierarchical %s is LOWER by %.4f (%.1f%%)", metric, delta, pct)
	default:
		t.Logf("RESULT: %s is EQUAL between flat and hierarchical", metric)
	}
}
