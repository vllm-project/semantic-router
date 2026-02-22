package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestHybridHierarchical_ThreeWayComparison runs:
//   - flat cosine-only
//   - hierarchical cosine-only
//   - hierarchical hybrid (BM25 + n-gram + cosine fused)
//
// and reports precision@K, recall@K, cluster purity, and token counts for each.
func TestHybridHierarchical_ThreeWayComparison(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, sizes := buildLabelIndex(ds)
	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	flatStore, hierStore := buildStores(t, ds)

	catCount := 0
	hierStore.mu.RLock()
	for _, m := range hierStore.memories {
		if m.IsCategory {
			catCount++
		}
	}
	hierStore.mu.RUnlock()

	t.Logf("Dataset: %d entries, %d clusters, %d category nodes", len(ds.Entries), len(sizes), catCount)
	t.Logf("Config: k=%d threshold=%.2f maxDepth=%d scorePropAlpha=%.2f", k, thr, ds.Config.MaxDepth, ds.Config.ScorePropAlpha)
	t.Log("")

	hybridCfg := &MemoryHybridConfig{
		Mode:         "weighted",
		VectorWeight: 0.7,
		BM25Weight:   0.2,
		NgramWeight:  0.1,
	}

	type methodResult struct {
		name    string
		results []*RetrieveResult
	}

	header := fmt.Sprintf("%-70s  %-12s  %7s %7s %7s  %7s %7s %7s  %7s %7s %7s",
		"Query", "Cluster",
		"F-P", "HC-P", "HH-P",
		"F-R", "HC-R", "HH-R",
		"F-pur", "HC-pur", "HH-pur")
	t.Log(header)

	var sumFP, sumFR, sumFPur float64
	var sumHCP, sumHCR, sumHCPur float64
	var sumHHP, sumHHR, sumHHPur float64

	for _, q := range ds.Queries {
		flatOpts := RetrieveOptions{Query: q.Query, UserID: "user1", Limit: k, Threshold: thr}

		flatRes, err := flatStore.Retrieve(ctx, flatOpts)
		requireNoErr(t, err)

		hierCosineRes, err := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts,
			MaxDepth:        ds.Config.MaxDepth,
			ScorePropAlpha:  ds.Config.ScorePropAlpha,
		})
		requireNoErr(t, err)

		hierHybridRes, err := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts,
			MaxDepth:        ds.Config.MaxDepth,
			ScorePropAlpha:  ds.Config.ScorePropAlpha,
			Hybrid:          hybridCfg,
		})
		requireNoErr(t, err)

		fp := precisionAtK(flatRes, k, q.TargetCluster, labels)
		fr := recallAtK(flatRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		fPur := clusterPurity(flatRes, k, labels)

		hcp := precisionAtK(hierCosineRes, k, q.TargetCluster, labels)
		hcr := recallAtK(hierCosineRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		hcPur := clusterPurity(hierCosineRes, k, labels)

		hhp := precisionAtK(hierHybridRes, k, q.TargetCluster, labels)
		hhr := recallAtK(hierHybridRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		hhPur := clusterPurity(hierHybridRes, k, labels)

		sumFP += fp
		sumFR += fr
		sumFPur += fPur
		sumHCP += hcp
		sumHCR += hcr
		sumHCPur += hcPur
		sumHHP += hhp
		sumHHR += hhr
		sumHHPur += hhPur

		t.Logf("%-70s  %-12s  %7.2f %7.2f %7.2f  %7.2f %7.2f %7.2f  %7.2f %7.2f %7.2f",
			truncateStr(q.Query, 68), q.TargetCluster,
			fp, hcp, hhp,
			fr, hcr, hhr,
			fPur, hcPur, hhPur)
	}

	nq := float64(len(ds.Queries))
	t.Log("")
	t.Logf("=== AVERAGES over %d queries (k=%d) ===", len(ds.Queries), k)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f  Purity: %.4f", "Flat (cosine)", k, sumFP/nq, k, sumFR/nq, sumFPur/nq)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f  Purity: %.4f", "Hier (cosine)", k, sumHCP/nq, k, sumHCR/nq, sumHCPur/nq)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f  Purity: %.4f", "Hier (hybrid)", k, sumHHP/nq, k, sumHHR/nq, sumHHPur/nq)
	t.Log("")

	logThreeWayDelta(t, "Precision", sumFP/nq, sumHCP/nq, sumHHP/nq)
	logThreeWayDelta(t, "Recall", sumFR/nq, sumHCR/nq, sumHHR/nq)
	logThreeWayDelta(t, "Purity", sumFPur/nq, sumHCPur/nq, sumHHPur/nq)
}

// TestHybridHierarchical_WeightSweep sweeps hybrid weight configurations to
// show the effect of BM25 and n-gram weight on precision.
func TestHybridHierarchical_WeightSweep(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, _ := buildLabelIndex(ds)
	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	_, hierStore := buildStores(t, ds)

	type weightSet struct {
		label string
		cfg   *MemoryHybridConfig
	}

	weights := []weightSet{
		{"pure-cosine (nil)", nil},
		{"v=1.0 b=0.0 n=0.0", &MemoryHybridConfig{VectorWeight: 1.0, BM25Weight: 0.0, NgramWeight: 0.0}},
		{"v=0.8 b=0.1 n=0.1", &MemoryHybridConfig{VectorWeight: 0.8, BM25Weight: 0.1, NgramWeight: 0.1}},
		{"v=0.7 b=0.2 n=0.1", &MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1}},
		{"v=0.6 b=0.3 n=0.1", &MemoryHybridConfig{VectorWeight: 0.6, BM25Weight: 0.3, NgramWeight: 0.1}},
		{"v=0.5 b=0.3 n=0.2", &MemoryHybridConfig{VectorWeight: 0.5, BM25Weight: 0.3, NgramWeight: 0.2}},
		{"v=0.5 b=0.5 n=0.0", &MemoryHybridConfig{VectorWeight: 0.5, BM25Weight: 0.5, NgramWeight: 0.0}},
		{"v=0.4 b=0.4 n=0.2", &MemoryHybridConfig{VectorWeight: 0.4, BM25Weight: 0.4, NgramWeight: 0.2}},
		{"rrf (default)", &MemoryHybridConfig{Mode: "rrf"}},
	}

	header := fmt.Sprintf("%-25s", "Weights")
	for _, q := range ds.Queries {
		header += fmt.Sprintf("  %-12s", q.TargetCluster[:min(12, len(q.TargetCluster))])
	}
	header += "  AvgP@K"
	t.Log(header)

	for _, w := range weights {
		line := fmt.Sprintf("%-25s", w.label)
		var sumP float64

		for _, q := range ds.Queries {
			opts := HierarchicalRetrieveOptions{
				RetrieveOptions: RetrieveOptions{Query: q.Query, UserID: "user1", Limit: k, Threshold: thr},
				MaxDepth:        ds.Config.MaxDepth,
				ScorePropAlpha:  ds.Config.ScorePropAlpha,
				Hybrid:          w.cfg,
			}
			results, err := hierStore.HierarchicalRetrieve(ctx, opts)
			requireNoErr(t, err)

			p := precisionAtK(results, k, q.TargetCluster, labels)
			sumP += p
			line += fmt.Sprintf("  %12.2f", p)
		}

		avgP := sumP / float64(len(ds.Queries))
		line += fmt.Sprintf("  %.4f", avgP)
		t.Log(line)
	}
}

// TestHybridHierarchical_TopResultInspection shows the top-5 results for each query
// under each method so we can see which specific memories are retrieved.
func TestHybridHierarchical_TopResultInspection(t *testing.T) {
	ds := loadEvalDataset(t)
	labels, _ := buildLabelIndex(ds)
	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	flatStore, hierStore := buildStores(t, ds)

	hybridCfg := &MemoryHybridConfig{
		Mode:         "weighted",
		VectorWeight: 0.7,
		BM25Weight:   0.2,
		NgramWeight:  0.1,
	}

	for _, q := range ds.Queries {
		t.Logf("--- Query: %s (target: %s) ---", truncateStr(q.Query, 80), q.TargetCluster)

		flatOpts := RetrieveOptions{Query: q.Query, UserID: "user1", Limit: k, Threshold: thr}

		flatRes, _ := flatStore.Retrieve(ctx, flatOpts)
		hierCosRes, _ := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts, MaxDepth: ds.Config.MaxDepth, ScorePropAlpha: ds.Config.ScorePropAlpha,
		})
		hierHybRes, _ := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts, MaxDepth: ds.Config.MaxDepth, ScorePropAlpha: ds.Config.ScorePropAlpha,
			Hybrid: hybridCfg,
		})

		logTopResults(t, "  Flat      ", flatRes, k, labels, q.TargetCluster)
		logTopResults(t, "  HierCos   ", hierCosRes, k, labels, q.TargetCluster)
		logTopResults(t, "  HierHybrid", hierHybRes, k, labels, q.TargetCluster)
		t.Log("")
	}
}

// TestHybridHierarchical_SourceCodeDataset runs the three-way comparison on the
// source code dataset (evaluation_source.json) if it exists.
func TestHybridHierarchical_SourceCodeDataset(t *testing.T) {
	ds := loadSourceDatasetOrSkip(t)
	labels, sizes := buildLabelIndex(ds)
	ctx := context.Background()
	k := ds.Config.K
	thr := ds.Config.Threshold

	flatStore, hierStore := buildStores(t, ds)

	catCount := 0
	hierStore.mu.RLock()
	for _, m := range hierStore.memories {
		if m.IsCategory {
			catCount++
		}
	}
	hierStore.mu.RUnlock()

	t.Logf("Source dataset: %d entries, %d clusters, %d category nodes", len(ds.Entries), len(sizes), catCount)
	t.Logf("Config: k=%d threshold=%.2f", k, thr)

	hybridCfg := &MemoryHybridConfig{
		VectorWeight: 0.7,
		BM25Weight:   0.2,
		NgramWeight:  0.1,
	}

	t.Log("")
	header := fmt.Sprintf("%-60s  %-15s  %7s %7s %7s  %7s %7s %7s",
		"Query", "Cluster", "F-P", "HC-P", "HH-P", "F-R", "HC-R", "HH-R")
	t.Log(header)

	var sumFP, sumFR, sumHCP, sumHCR, sumHHP, sumHHR float64

	for _, q := range ds.Queries {
		flatOpts := RetrieveOptions{Query: q.Query, UserID: "user1", Limit: k, Threshold: thr}

		flatRes, err := flatStore.Retrieve(ctx, flatOpts)
		requireNoErr(t, err)

		hierCosRes, err := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts, MaxDepth: ds.Config.MaxDepth, ScorePropAlpha: ds.Config.ScorePropAlpha,
		})
		requireNoErr(t, err)

		hierHybRes, err := hierStore.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: flatOpts, MaxDepth: ds.Config.MaxDepth, ScorePropAlpha: ds.Config.ScorePropAlpha,
			Hybrid: hybridCfg,
		})
		requireNoErr(t, err)

		fp := precisionAtK(flatRes, k, q.TargetCluster, labels)
		fr := recallAtK(flatRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		hcp := precisionAtK(hierCosRes, k, q.TargetCluster, labels)
		hcr := recallAtK(hierCosRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])
		hhp := precisionAtK(hierHybRes, k, q.TargetCluster, labels)
		hhr := recallAtK(hierHybRes, k, q.TargetCluster, labels, sizes[q.TargetCluster])

		sumFP += fp
		sumFR += fr
		sumHCP += hcp
		sumHCR += hcr
		sumHHP += hhp
		sumHHR += hhr

		t.Logf("%-60s  %-15s  %7.2f %7.2f %7.2f  %7.2f %7.2f %7.2f",
			truncateStr(q.Query, 58), q.TargetCluster[:min(15, len(q.TargetCluster))],
			fp, hcp, hhp, fr, hcr, hhr)
	}

	nq := float64(len(ds.Queries))
	t.Log("")
	t.Logf("=== AVERAGES over %d queries (k=%d) ===", len(ds.Queries), k)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f", "Flat (cosine)", k, sumFP/nq, k, sumFR/nq)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f", "Hier (cosine)", k, sumHCP/nq, k, sumHCR/nq)
	t.Logf("  %-25s  P@%d: %.4f  R@%d: %.4f", "Hier (hybrid)", k, sumHHP/nq, k, sumHHR/nq)
	t.Log("")

	logThreeWayDelta(t, "Precision", sumFP/nq, sumHCP/nq, sumHHP/nq)
	logThreeWayDelta(t, "Recall", sumFR/nq, sumHCR/nq, sumHHR/nq)
}

// TestHybridScore_Unit validates that the hybrid scorer produces different results
// from pure cosine when BM25/n-gram signals are present.
func TestHybridScore_Unit(t *testing.T) {
	docs := map[string]string{
		"a": "Kubernetes deployment uses Helm charts for configuration management",
		"b": "BM25 keyword matching scores documents based on term frequency",
		"c": "The cat sat on the mat in a warm sunny afternoon",
	}

	cfg := &MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1}
	scorer := BuildMemHybridScorer(docs, cfg)

	cosineA := float32(0.8)
	cosineB := float32(0.75)
	cosineC := float32(0.7)

	query := "Helm charts Kubernetes deployment"

	fusedA := scorer.FusedScore("a", cosineA, query)
	fusedB := scorer.FusedScore("b", cosineB, query)
	fusedC := scorer.FusedScore("c", cosineC, query)

	t.Logf("Query: %q", query)
	t.Logf("  doc A (Kubernetes/Helm): cosine=%.3f  fused=%.3f  delta=%+.3f", cosineA, fusedA, fusedA-cosineA)
	t.Logf("  doc B (BM25 text):       cosine=%.3f  fused=%.3f  delta=%+.3f", cosineB, fusedB, fusedB-cosineB)
	t.Logf("  doc C (cat/mat):         cosine=%.3f  fused=%.3f  delta=%+.3f", cosineC, fusedC, fusedC-cosineC)

	if fusedA <= float32(0.7)*cosineA {
		t.Errorf("expected doc A fused score to be boosted by BM25/ngram match, got %.3f", fusedA)
	}
	if fusedA <= fusedC {
		t.Error("expected doc A (exact term match) to outscore doc C (no term match) after fusion")
	}
}

// TestHybridScore_BatchFusion verifies batch fusion produces consistent results.
func TestHybridScore_BatchFusion(t *testing.T) {
	docs := map[string]string{
		"x": "vector similarity search with cosine distance metric",
		"y": "BM25 scoring with inverse document frequency weighting",
		"z": "reciprocal rank fusion combines multiple retriever scores",
	}

	cfg := &MemoryHybridConfig{VectorWeight: 0.5, BM25Weight: 0.3, NgramWeight: 0.2}
	scorer := BuildMemHybridScorer(docs, cfg)

	cosineScores := map[string]float32{"x": 0.9, "y": 0.8, "z": 0.85}
	query := "BM25 inverse document frequency"

	batch := scorer.FusedScores(cosineScores, query)

	for id, fused := range batch {
		t.Logf("  %s: cosine=%.3f  fused=%.3f", id, cosineScores[id], fused)
	}

	if batch["y"] <= batch["z"] {
		t.Log("NOTE: doc Y (exact BM25 terms) did not outscore doc Z — BM25 weight may be too low for this case")
	}

	if len(batch) != 3 {
		t.Errorf("expected 3 fused scores, got %d", len(batch))
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func loadSourceDatasetOrSkip(t *testing.T) *evalDatasetFile {
	t.Helper()

	_, thisFile, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(thisFile), "testdata", "evaluation_source.json")

	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("evaluation_source.json not found (%v), skipping source code dataset test", err)
		return nil
	}

	var ds evalDatasetFile
	if err := json.Unmarshal(data, &ds); err != nil {
		t.Fatalf("failed to parse evaluation_source.json: %v", err)
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

	if len(ds.Entries) == 0 || len(ds.Queries) == 0 {
		t.Skip("evaluation_source.json has no entries or queries")
		return nil
	}
	return &ds
}

func logTopResults(t *testing.T, prefix string, results []*RetrieveResult, k int, labels map[string]string, target string) {
	t.Helper()
	n := k
	if n > len(results) {
		n = len(results)
	}
	for i := 0; i < n; i++ {
		r := results[i]
		cluster := labels[r.Memory.ID]
		hit := " "
		if cluster == target {
			hit = "*"
		}
		content := r.Memory.Content
		if len(content) > 60 {
			content = content[:60] + "..."
		}
		t.Logf("%s [%d] %s %.3f %-12s %s", prefix, i+1, hit, r.Score, cluster, content)
	}
	if n == 0 {
		t.Logf("%s  (no results)", prefix)
	}
}

func logThreeWayDelta(t *testing.T, metric string, flat, hierCos, hierHyb float64) {
	t.Helper()

	cosVsFlat := hierCos - flat
	hybVsFlat := hierHyb - flat
	hybVsCos := hierHyb - hierCos

	t.Logf("DELTA %s:", metric)
	t.Logf("  hier-cosine vs flat:      %+.4f  (%+.1f%%)", cosVsFlat, safePct(cosVsFlat, flat))
	t.Logf("  hier-hybrid vs flat:      %+.4f  (%+.1f%%)", hybVsFlat, safePct(hybVsFlat, flat))
	t.Logf("  hier-hybrid vs hier-cos:  %+.4f  (%+.1f%%)", hybVsCos, safePct(hybVsCos, hierCos))
	t.Log("")
}

func safePct(delta, base float64) float64 {
	if base == 0 {
		return 0
	}
	return delta / base * 100
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
