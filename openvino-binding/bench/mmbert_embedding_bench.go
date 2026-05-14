package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	openvino "github.com/vllm-project/semantic-router/openvino-binding"
)

type BenchCfg struct {
	Name        string
	Text        string
	Concurrency int
	Iterations  int
}

type BenchRes struct {
	Backend    string
	Cfg        BenchCfg
	Latencies  []time.Duration
	Mean       time.Duration
	Median     time.Duration
	P95        time.Duration
	P99        time.Duration
	Min        time.Duration
	Max        time.Duration
	Throughput float64
	Errs       int
}

func main() {
	ovModel := getenv("OPENVINO_MODEL_PATH")
	candleModel := getenv("CANDLE_MODEL_PATH")
	if ovModel == "" || candleModel == "" {
		fmt.Println("OPENVINO_MODEL_PATH and CANDLE_MODEL_PATH are required")
		os.Exit(1)
	}

	// NOTE: OpenVINO maxLength is tokenizer sequence length, not embedding dimension.
	ovMaxLength := getenvInt("OV_MAX_LENGTH", 512)
	candleTargetDim := getenvInt("CANDLE_TARGET_DIM", 768)
	minCosine := getenvFloat("EMBEDDING_MIN_COSINE", 0.90)
	stageTimingEnabled := getenvBool("EMBEDDING_STAGE_TIMING", false)
	stageTimingSamples := getenvInt("EMBEDDING_STAGE_TIMING_SAMPLES", 10)
	lengthProfile := normalizeLengthProfile(getenv("EMBEDDING_LENGTH_PROFILE"))

	fmt.Println("== mmBERT Embedding Benchmark (OpenVINO vs Candle) ==")
	fmt.Println("OPENVINO_MODEL_PATH=", ovModel)
	fmt.Println("CANDLE_MODEL_PATH=", candleModel)
	fmt.Println("OV_MAX_LENGTH=", ovMaxLength)
	fmt.Println("CANDLE_TARGET_DIM=", candleTargetDim)
	fmt.Println("EMBEDDING_MIN_COSINE=", minCosine)
	fmt.Println("EMBEDDING_STAGE_TIMING=", stageTimingEnabled)
	fmt.Println("EMBEDDING_STAGE_TIMING_SAMPLES=", stageTimingSamples)
	fmt.Println("EMBEDDING_LENGTH_PROFILE=", lengthProfile)

	if err := openvino.InitModernBertEmbedding(ovModel, "CPU"); err != nil {
		fmt.Println("openvino init failed:", err)
		os.Exit(1)
	}
	if err := candle.InitEmbeddingModels("", "", candleModel, true); err != nil {
		fmt.Println("candle init failed:", err)
		os.Exit(1)
	}

	fmt.Println("Verifying embedding correctness (required before benchmark)...")
	if err := verifyEmbeddingCorrectness(ovMaxLength, candleTargetDim, minCosine); err != nil {
		fmt.Println("embedding correctness check failed:", err)
		os.Exit(1)
	}
	fmt.Println("Embedding correctness check passed")

	cfgs := buildEmbeddingConfigs(lengthProfile)

	allResults := make([]BenchRes, 0, len(cfgs)*2)

	for _, c := range cfgs {
		logOpenVINOTruncationIfAny(c, ovMaxLength)
		ov := run(c, "openvino", func(text string) error {
			_, err := openvino.GetModernBertEmbedding(text, ovMaxLength)
			return err
		})
		ca := run(c, "candle", func(text string) error {
			_, err := candle.GetEmbeddingWithModelType(text, "mmbert", candleTargetDim)
			return err
		})
		allResults = append(allResults, ov, ca)
		printRes(ov)
		if stageTimingEnabled {
			printStageTimingProbe(
				"openvino",
				c,
				stageTimingSamples,
				func(text string) error {
					_, err := openvino.TokenizeText(text, ovMaxLength)
					return err
				},
				func(text string) ([]float32, error) {
					return openvino.GetModernBertEmbedding(text, ovMaxLength)
				},
			)
		}
		printRes(ca)
		if stageTimingEnabled {
			printStageTimingProbe(
				"candle",
				c,
				stageTimingSamples,
				func(text string) error {
					_, err := candle.TokenizeText(text, ovMaxLength)
					return err
				},
				func(text string) ([]float32, error) {
					out, err := candle.GetEmbeddingWithModelType(text, "mmbert", candleTargetDim)
					if err != nil {
						return nil, err
					}
					return out.Embedding, nil
				},
			)
		}

		if ov.Mean <= 0 || ca.Mean <= 0 {
			fmt.Printf("speedup(openvino/candle) cfg=%s : n/a (insufficient successful samples)\n\n", c.Name)
			continue
		}
		fmt.Printf("speedup(openvino/candle) cfg=%s : %.2fx (higher is better)\n\n", c.Name, float64(ca.Mean)/float64(ov.Mean))
	}

	printSummaryTable(cfgs, allResults)
}

func printStageTimingProbe(
	backend string,
	cfg BenchCfg,
	samples int,
	tokenizeFn func(string) error,
	embedFn func(string) ([]float32, error),
) {
	if samples <= 0 {
		samples = 1
	}

	tokDur := make([]time.Duration, 0, samples)
	apiDur := make([]time.Duration, 0, samples)
	postDur := make([]time.Duration, 0, samples)
	errCount := 0
	tokenizeUnavailable := false

	for i := 0; i < samples; i++ {
		if !tokenizeUnavailable {
			t0 := time.Now()
			err := tokenizeFn(cfg.Text)
			tokDur = append(tokDur, time.Since(t0))
			if err != nil {
				tokenizeUnavailable = true
				errCount++
			}
		}

		t1 := time.Now()
		emb, err := embedFn(cfg.Text)
		apiDur = append(apiDur, time.Since(t1))
		if err != nil {
			errCount++
			continue
		}

		t2 := time.Now()
		_ = l2Norm(emb)
		postDur = append(postDur, time.Since(t2))
	}

	fmt.Printf(
		"stage_timing backend=%s cfg=%s samples=%d errs=%d tok_mean=%s tok_p95=%s api_mean=%s api_p95=%s post_mean=%s post_p95=%s note=%q\n",
		backend,
		cfg.Name,
		samples,
		errCount,
		formatDurationOrNA(tokDur, tokenizeUnavailable),
		formatDurationPctlOrNA(tokDur, 0.95, tokenizeUnavailable),
		formatDurationOrNA(apiDur, false),
		formatDurationPctlOrNA(apiDur, 0.95, false),
		formatDurationOrNA(postDur, false),
		formatDurationPctlOrNA(postDur, 0.95, false),
		"diagnostic probe; stage values are standalone micro-timings, not additive",
	)
}

func verifyEmbeddingCorrectness(ovMaxLength int, candleTargetDim int, minCosine float64) error {
	testTexts := []string{
		"hello semantic router",
		"this is a short text for embedding correctness",
		"please redact PII entities such as names and phone numbers",
		"OpenVINO and Candle should generate semantically consistent vectors",
	}

	failures := 0
	cosSum := 0.0
	checked := 0

	for i, text := range testTexts {
		ovEmb, err := openvino.GetModernBertEmbedding(text, ovMaxLength)
		if err != nil {
			return fmt.Errorf("openvino embedding failed on test %d: %w", i+1, err)
		}

		candleOut, err := candle.GetEmbeddingWithModelType(text, "mmbert", candleTargetDim)
		if err != nil {
			return fmt.Errorf("candle embedding failed on test %d: %w", i+1, err)
		}

		caEmb := candleOut.Embedding
		if len(ovEmb) == 0 || len(caEmb) == 0 {
			return fmt.Errorf("empty embedding on test %d: ov_dim=%d candle_dim=%d", i+1, len(ovEmb), len(caEmb))
		}

		common := minInt(len(ovEmb), len(caEmb))
		if common < 64 {
			return fmt.Errorf("embedding dimension too small to compare on test %d: common_dim=%d", i+1, common)
		}

		cos := cosineSimilarity(ovEmb[:common], caEmb[:common])
		checked++
		cosSum += cos
		if cos < minCosine {
			failures++
			fmt.Printf("  embedding mismatch test=%d cosine=%.5f threshold=%.5f ov_dim=%d candle_dim=%d\n", i+1, cos, minCosine, len(ovEmb), len(caEmb))
		}
	}

	avgCos := cosSum / float64(maxInt(checked, 1))
	fmt.Printf("  embedding correctness summary: checked=%d failures=%d avg_cosine=%.5f\n", checked, failures, avgCos)
	if failures > 0 {
		return fmt.Errorf("%d/%d embedding checks below cosine threshold %.5f", failures, checked, minCosine)
	}

	return nil
}

func logOpenVINOTruncationIfAny(cfg BenchCfg, ovMaxLength int) {
	if ovMaxLength <= 0 {
		return
	}

	probeMaxLength := maxInt(ovMaxLength*2, 2048)
	if probeMaxLength <= ovMaxLength {
		probeMaxLength = ovMaxLength + 1
	}

	probe, err := openvino.TokenizeText(cfg.Text, probeMaxLength)
	if err != nil {
		fmt.Printf("note: cfg=%s truncation probe skipped (tokenize failed: %v)\n", cfg.Name, err)
		return
	}

	if len(probe.TokenIDs) > ovMaxLength {
		fmt.Printf("note: cfg=%s text token length=%d exceeds OV_MAX_LENGTH=%d; input will be truncated before embedding\n", cfg.Name, len(probe.TokenIDs), ovMaxLength)
	}
}

func run(cfg BenchCfg, backend string, fn func(text string) error) BenchRes {
	res := BenchRes{Backend: backend, Cfg: cfg, Latencies: make([]time.Duration, 0, cfg.Concurrency*cfg.Iterations)}
	var wg sync.WaitGroup
	var mu sync.Mutex
	start := time.Now()
	for i := 0; i < cfg.Concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < cfg.Iterations; j++ {
				t0 := time.Now()
				err := fn(cfg.Text)
				dt := time.Since(t0)
				mu.Lock()
				if err != nil {
					res.Errs++
				} else {
					res.Latencies = append(res.Latencies, dt)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)
	if len(res.Latencies) == 0 {
		return res
	}
	sort.Slice(res.Latencies, func(i, j int) bool { return res.Latencies[i] < res.Latencies[j] })
	res.Min = res.Latencies[0]
	res.Max = res.Latencies[len(res.Latencies)-1]
	res.Median = res.Latencies[len(res.Latencies)/2]
	p95Idx := int(float64(len(res.Latencies)) * 0.95)
	if p95Idx >= len(res.Latencies) {
		p95Idx = len(res.Latencies) - 1
	}
	res.P95 = res.Latencies[p95Idx]
	p99Idx := int(float64(len(res.Latencies)) * 0.99)
	if p99Idx >= len(res.Latencies) {
		p99Idx = len(res.Latencies) - 1
	}
	res.P99 = res.Latencies[p99Idx]
	var sum time.Duration
	for _, v := range res.Latencies {
		sum += v
	}
	res.Mean = sum / time.Duration(len(res.Latencies))
	res.Throughput = float64(len(res.Latencies)) / elapsed.Seconds()
	return res
}

func printRes(r BenchRes) {
	fmt.Printf("backend=%s cfg=%s n=%d errs=%d mean=%v p50=%v p95=%v p99=%v min=%v max=%v tps=%.2f\n",
		r.Backend, r.Cfg.Name, len(r.Latencies), r.Errs, r.Mean, r.Median, r.P95, r.P99, r.Min, r.Max, r.Throughput)
}

func cosineSimilarity(a []float32, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := minInt(len(a), len(b))
	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func l2Norm(v []float32) float64 {
	if len(v) == 0 {
		return 0
	}
	var sum float64
	for _, x := range v {
		fx := float64(x)
		sum += fx * fx
	}
	return math.Sqrt(sum)
}

func formatDurationOrNA(v []time.Duration, unavailable bool) string {
	if unavailable || len(v) == 0 {
		return "n/a"
	}
	var sum time.Duration
	for _, d := range v {
		sum += d
	}
	return (sum / time.Duration(len(v))).String()
}

func formatDurationPctlOrNA(v []time.Duration, p float64, unavailable bool) string {
	if unavailable || len(v) == 0 {
		return "n/a"
	}
	buf := append([]time.Duration(nil), v...)
	sort.Slice(buf, func(i, j int) bool { return buf[i] < buf[j] })
	idx := int(float64(len(buf)) * p)
	if idx >= len(buf) {
		idx = len(buf) - 1
	}
	if idx < 0 {
		idx = 0
	}
	return buf[idx].String()
}

func getenv(k string) string {
	v := os.Getenv(k)
	if v == "" {
		return ""
	}
	return v
}

func getenvInt(k string, def int) int {
	raw := os.Getenv(k)
	if raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		fmt.Printf("invalid %s=%q, using default %d\n", k, raw, def)
		return def
	}
	return v
}

func getenvFloat(k string, def float64) float64 {
	raw := os.Getenv(k)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		fmt.Printf("invalid %s=%q, using default %.4f\n", k, raw, def)
		return def
	}
	return v
}

func getenvBool(k string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(k)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		fmt.Printf("invalid %s=%q, using default %t\n", k, raw, def)
		return def
	}
}

func printSummaryTable(cfgs []BenchCfg, results []BenchRes) {
	type pair struct {
		ov *BenchRes
		ca *BenchRes
	}

	pairs := make(map[string]*pair)
	for i := range results {
		r := &results[i]
		p, ok := pairs[r.Cfg.Name]
		if !ok {
			p = &pair{}
			pairs[r.Cfg.Name] = p
		}
		switch strings.ToLower(r.Backend) {
		case "openvino":
			p.ov = r
		case "candle":
			p.ca = r
		}
	}

	fmt.Println("=== Final Summary Table ===")
	fmt.Printf("%-10s %-4s %-6s %-11s %-10s %-10s %-6s %-11s %-10s %-10s %-8s\n",
		"cfg", "conc", "ov_n", "ov_mean_ms", "ov_p95_ms", "ov_tps", "ca_n", "ca_mean_ms", "ca_p95_ms", "ca_tps", "speedup")
	fmt.Println(strings.Repeat("-", 108))

	for _, cfg := range cfgs {
		p, ok := pairs[cfg.Name]
		if !ok || p.ov == nil || p.ca == nil {
			fmt.Printf("%-10s %-4d %-6s %-11s %-10s %-10s %-6s %-11s %-10s %-10s %-8s\n",
				cfg.Name, cfg.Concurrency, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a")
			continue
		}

		speedup := "n/a"
		if p.ov.Mean > 0 && p.ca.Mean > 0 {
			speedup = fmt.Sprintf("%.2fx", float64(p.ca.Mean)/float64(p.ov.Mean))
		}

		fmt.Printf("%-10s %-4d %-6d %-11.2f %-10.2f %-10.2f %-6d %-11.2f %-10.2f %-10.2f %-8s\n",
			cfg.Name,
			cfg.Concurrency,
			len(p.ov.Latencies),
			float64(p.ov.Mean.Microseconds())/1000.0,
			float64(p.ov.P95.Microseconds())/1000.0,
			p.ov.Throughput,
			len(p.ca.Latencies),
			float64(p.ca.Mean.Microseconds())/1000.0,
			float64(p.ca.P95.Microseconds())/1000.0,
			p.ca.Throughput,
			speedup,
		)
	}

	fmt.Println()
}

func normalizeLengthProfile(raw string) string {
	v := strings.TrimSpace(strings.ToLower(raw))
	if v == "" {
		return "mixed"
	}
	switch v {
	case "mixed", "fixed-32", "fixed-128", "fixed-512", "fixed-1024", "fixed-2048":
		return v
	default:
		fmt.Printf("invalid EMBEDDING_LENGTH_PROFILE=%q, using default mixed\n", raw)
		return "mixed"
	}
}

func buildEmbeddingConfigs(lengthProfile string) []BenchCfg {
	switch lengthProfile {
	case "fixed-32":
		text := buildFixedLengthText(32)
		return []BenchCfg{
			{Name: "fixed32-c1", Text: text, Concurrency: 1, Iterations: 40},
			{Name: "fixed32-c8", Text: text, Concurrency: 8, Iterations: 40},
		}
	case "fixed-128":
		text := buildFixedLengthText(128)
		return []BenchCfg{
			{Name: "fixed128-c1", Text: text, Concurrency: 1, Iterations: 40},
			{Name: "fixed128-c8", Text: text, Concurrency: 8, Iterations: 40},
		}
	case "fixed-512":
		text := buildFixedLengthText(512)
		return []BenchCfg{
			{Name: "fixed512-c1", Text: text, Concurrency: 1, Iterations: 40},
			{Name: "fixed512-c8", Text: text, Concurrency: 8, Iterations: 40},
		}
	case "fixed-1024":
		text := buildFixedLengthText(1024)
		return []BenchCfg{
			{Name: "fixed1024-c1", Text: text, Concurrency: 1, Iterations: 40},
			{Name: "fixed1024-c8", Text: text, Concurrency: 8, Iterations: 40},
		}
	case "fixed-2048":
		text := buildFixedLengthText(2048)
		return []BenchCfg{
			{Name: "fixed2048-c1", Text: text, Concurrency: 1, Iterations: 40},
			{Name: "fixed2048-c8", Text: text, Concurrency: 8, Iterations: 40},
		}
	default:
		return []BenchCfg{
			{Name: "short-c1", Text: "hello semantic router", Concurrency: 1, Iterations: 40},
			{Name: "short-c8", Text: "hello semantic router", Concurrency: 8, Iterations: 40},
			{Name: "long-c1", Text: "This is a longer sentence for measuring mmbert embedding latency after switching backend to openvino.", Concurrency: 1, Iterations: 40},
		}
	}
}

func buildFixedLengthText(tokens int) string {
	if tokens <= 0 {
		tokens = 1
	}

	subjects := []string{"A customer", "The support agent", "Our policy engine", "The checkout service", "A data analyst", "The mobile client", "The routing layer", "An internal tool"}
	verbs := []string{"reports", "requests", "validates", "compares", "aggregates", "routes", "summarizes", "checks"}
	objects := []string{"a delayed refund", "an account recovery issue", "a suspicious login attempt", "an address update request", "a pricing mismatch", "a failed payment", "a shipment status change", "a compliance warning"}
	contexts := []string{"during peak traffic in production", "after a deployment to staging", "for a multilingual user query", "with strict latency constraints", "under a high-concurrency workload", "while preserving auditability", "before escalating to human review", "for a cross-region request"}

	words := make([]string, 0, tokens+16)
	for i := 0; len(words) < tokens; i++ {
		sentence := fmt.Sprintf("%s %s %s %s.",
			subjects[i%len(subjects)],
			verbs[(i*3+1)%len(verbs)],
			objects[(i*5+2)%len(objects)],
			contexts[(i*7+3)%len(contexts)],
		)
		parts := strings.Fields(sentence)
		remaining := tokens - len(words)
		if len(parts) > remaining {
			parts = parts[:remaining]
		}
		words = append(words, parts...)
	}

	text := strings.Join(words, " ")
	if text == "" {
		return "semantic"
	}
	if !strings.HasSuffix(text, ".") {
		text += "."
	}
	return text
}
