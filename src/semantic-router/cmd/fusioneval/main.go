// Command fusioneval is the paired, multi-arm evaluation driver for
// grounding-aware Fusion. It generates the panel ONCE per item and caches it,
// then synthesizes every arm from the byte-identical cached panel so the deltas
// isolate the intervention (no per-arm panel-regeneration noise — the flaw that
// made the first DRACO A/B uninterpretable, see bench/grounded_fusion/FINDINGS.md).
//
// Arms:
//
//	A  judge-solo      one model, no fusion          — does fusion beat one model?
//	B  plain-fusion    grounding disabled            — does grounding add anything?
//	C  weight          shipped default               — the actual claim
//	D  placebo         weight on seeded-random scores — score signal vs any weighting
//	annotate / filter  optional, behind --arms
//
// It wires the REAL candle NLI (the same scorer the router uses) so arm C/D
// reflect the shipped path. Grading + the KEEP/KILL verdict live in Python
// (bench/grounded_fusion/grade_only.py + compare_multiarm.py), which read the
// answers_{arm}.jsonl this driver emits.
//
// Build/run (needs the candle lib + NLI model + a running Ollama):
//
//	cd src/semantic-router && \
//	  CGO_LDFLAGS="-L$PWD/../../candle-binding/target/release" go build -o ../../bin/fusioneval ./cmd/fusioneval
//	LD_LIBRARY_PATH=$PWD/../../candle-binding/target/release ../../bin/fusioneval \
//	  --items items.jsonl --nli-model models/mom-halugate-explainer \
//	  --endpoint http://localhost:11435/v1/chat/completions \
//	  --judge qwen3:14b --panel qwen3:8b,llama3.1:8b,gemma3:12b \
//	  --arms A,B,C,D --out-dir results
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/openai/openai-go"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

type options struct {
	itemsPath   string
	panelCache  string
	outDir      string
	arms        []string
	judge       string
	panelModels []string
	endpoint    string
	nliModel    string
	useCPU      bool
	seed        int64
	placeboSeed uint64
	reference   string
	temperature float64
	maxTokens   int
	maxItems    int
}

// item is one evaluation question. The Python side (datasets.py) owns DRACO
// parsing and dumps these via `python -m bench.grounded_fusion.items`.
type item struct {
	ID       string `json:"id"`
	Domain   string `json:"domain"`
	Question string `json:"question"`
	Context  string `json:"context,omitempty"`
}

type usageRec struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type cachedResponse struct {
	Model     string   `json:"model"`
	Content   string   `json:"content"`
	Reasoning string   `json:"reasoning,omitempty"`
	Usage     usageRec `json:"usage"`
}

type cachedPanelItem struct {
	ItemID      string           `json:"item_id"`
	Question    string           `json:"question"`
	Context     string           `json:"context,omitempty"`
	PanelSeed   int64            `json:"panel_seed"`
	Panel       []cachedResponse `json:"panel"`
	PanelSHA256 string           `json:"panel_sha256"`
}

type panelRec struct {
	Model          string   `json:"model"`
	GroundingScore *float64 `json:"grounding_score"`
	Dropped        bool     `json:"dropped"`
	Flagged        []string `json:"flagged"`
}

// answerRecord is the pre-grading output consumed by grade_only.py.
type answerRecord struct {
	ID               string     `json:"id"`
	Domain           string     `json:"domain"`
	Arm              string     `json:"arm"`
	FinalAnswer      string     `json:"final_answer"`
	Usage            usageRec   `json:"usage"`
	GroundingPresent bool       `json:"grounding_present"`
	ReferenceMode    string     `json:"reference_mode,omitempty"`
	Policy           string     `json:"policy,omitempty"`
	Panel            []panelRec `json:"panel"`
	PanelSHA256      string     `json:"panel_sha256"`
	Error            string     `json:"error,omitempty"`
}

func main() {
	opt := parseFlags()
	if err := run(opt); err != nil {
		fmt.Fprintln(os.Stderr, "fusioneval:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	var (
		arms  = flag.String("arms", "A,B,C,D", "comma-separated arms to run: A(judge-solo) B(plain) C(weight) D(placebo) annotate filter")
		panel = flag.String("panel", "qwen3:8b,llama3.1:8b,gemma3:12b", "comma-separated panel (analysis) models")
	)
	opt := options{}
	flag.StringVar(&opt.itemsPath, "items", "", "path to items JSONL ({id,domain,question,context}) — dump via bench.grounded_fusion.items")
	flag.StringVar(&opt.panelCache, "panel-cache", "", "panel cache JSONL path (default <out-dir>/panel_cache.jsonl)")
	flag.StringVar(&opt.outDir, "out-dir", "results", "output directory for answers_{arm}.jsonl")
	flag.StringVar(&opt.judge, "judge", "qwen3:14b", "judge / synthesis model")
	flag.StringVar(&opt.endpoint, "endpoint", "http://localhost:11435/v1/chat/completions", "OpenAI-compatible chat endpoint (Ollama proxy)")
	flag.StringVar(&opt.nliModel, "nli-model", "models/mom-halugate-explainer", "candle NLI model path for panel-mode grounding")
	flag.BoolVar(&opt.useCPU, "use-cpu", true, "run the candle NLI model on CPU")
	flag.Int64Var(&opt.seed, "seed", 42, "panel generation seed (recorded; determinism depends on the backend)")
	flag.Uint64Var(&opt.placeboSeed, "placebo-seed", 7, "base seed for the random-weight placebo (arm D)")
	flag.StringVar(&opt.reference, "grounding-reference", config.FusionGroundingReferencePanel, "grounding reference mode: panel|context|hybrid")
	flag.Float64Var(&opt.temperature, "temperature", 0, "sampling temperature for all calls")
	flag.IntVar(&opt.maxTokens, "max-tokens", 0, "max completion tokens (0 = backend default)")
	flag.IntVar(&opt.maxItems, "max-items", 0, "cap items processed (0 = all); use for a smoke run")
	flag.Parse()

	opt.arms = splitCSV(*arms)
	opt.panelModels = splitCSV(*panel)
	if opt.panelCache == "" {
		opt.panelCache = filepath.Join(opt.outDir, "panel_cache.jsonl")
	}
	return opt
}

func run(opt options) error {
	if opt.itemsPath == "" {
		return fmt.Errorf("--items is required")
	}
	if err := os.MkdirAll(opt.outDir, 0o755); err != nil {
		return err
	}

	items, err := loadItems(opt.itemsPath)
	if err != nil {
		return fmt.Errorf("load items: %w", err)
	}
	if opt.maxItems > 0 && len(items) > opt.maxItems {
		items = items[:opt.maxItems]
	}

	// Wire the REAL candle NLI for panel-mode grounding (arms C/D's reference).
	// Context mode would additionally need the hallucination detector; deferred.
	if err = candle.InitNLIModel(opt.nliModel, opt.useCPU); err != nil {
		return fmt.Errorf("init candle NLI model %q: %w", opt.nliModel, err)
	}
	looper.SetGroundingBackends(realNLI(), nil)

	looperCfg := &config.LooperConfig{Endpoint: opt.endpoint}
	client := looper.NewClient(looperCfg)
	fusion := looper.NewFusionLooper(looperCfg)

	// Phase 1: build (or resume) the panel cache — generate each panel ONCE.
	cache, err := buildPanelCache(client, opt, items)
	if err != nil {
		return fmt.Errorf("build panel cache: %w", err)
	}
	fmt.Printf("panel cache ready: %d items at %s\n", len(cache), opt.panelCache)

	// Phase 2: per arm, synthesize every item from its cached panel.
	for _, arm := range opt.arms {
		if err := runArm(fusion, client, opt, looperCfg, items, cache, arm); err != nil {
			return fmt.Errorf("arm %s: %w", arm, err)
		}
	}
	fmt.Println("done. grade with: python -m bench.grounded_fusion.grade_only ...")
	return nil
}

func realNLI() looper.NLIClassifyFunc {
	return func(premise, hypothesis string) (float32, float32, error) {
		r, err := candle.ClassifyNLI(premise, hypothesis)
		if err != nil {
			return 0, 0, err
		}
		return r.EntailmentProb, r.ContradictProb, nil
	}
}

// placeboNLI returns deterministic seeded-random scores: reproducible for a given
// (seed, premise, hypothesis) but carrying no real signal. Mirrors the in-package
// test placebo so arm D weights on noise, isolating the score's signal.
func placeboNLI(seed uint64) looper.NLIClassifyFunc {
	return func(premise, hypothesis string) (float32, float32, error) {
		h := fnv.New64a()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], seed)
		_, _ = h.Write(b[:])
		_, _ = h.Write([]byte(premise))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(hypothesis))
		// Deterministic seed for the placebo; the uint64->int64 wrap is harmless.
		r := rand.New(rand.NewSource(int64(h.Sum64()))) //nolint:gosec // deterministic eval seed, overflow harmless
		entail := float32(r.Float64())
		contradict := float32(r.Float64() * (1 - float64(entail)))
		return entail, contradict, nil
	}
}

// buildPanelCache generates any panel not already cached, appending to the cache
// file. Keyed on item ID; safe to re-run (resume).
func buildPanelCache(client *looper.Client, opt options, items []item) (map[string]cachedPanelItem, error) {
	cache, err := loadPanelCache(opt.panelCache)
	if err != nil {
		return nil, err
	}
	fh, err := os.OpenFile(opt.panelCache, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	for _, it := range items {
		if _, ok := cache[it.ID]; ok {
			continue
		}
		panel, err := generatePanel(client, opt, it)
		if err != nil {
			return nil, fmt.Errorf("generate panel for %s: %w", it.ID, err)
		}
		entry := cachedPanelItem{
			ItemID:      it.ID,
			Question:    it.Question,
			Context:     it.Context,
			PanelSeed:   opt.seed,
			Panel:       panel,
			PanelSHA256: panelSHA256(panel),
		}
		if err := appendJSON(fh, entry); err != nil {
			return nil, err
		}
		cache[it.ID] = entry
		fmt.Printf("  panel cached: %s (%d responses)\n", it.ID, len(panel))
	}
	return cache, nil
}

func generatePanel(client *looper.Client, opt options, it item) ([]cachedResponse, error) {
	out := make([]cachedResponse, 0, len(opt.panelModels))
	for _, model := range opt.panelModels {
		req := buildRequest(it.Question, it.Context, opt)
		resp, err := client.CallModel(context.Background(), req, model, false, 1, nil, "")
		if err != nil {
			return nil, fmt.Errorf("model %q: %w", model, err)
		}
		out = append(out, cachedResponse{
			Model:     model,
			Content:   resp.Content,
			Reasoning: resp.ReasoningContent,
			Usage:     usageRec(resp.Usage),
		})
	}
	return out, nil
}

func runArm(
	fusion *looper.FusionLooper,
	client *looper.Client,
	opt options,
	looperCfg *config.LooperConfig,
	items []item,
	cache map[string]cachedPanelItem,
	arm string,
) error {
	outPath := filepath.Join(opt.outDir, fmt.Sprintf("answers_%s.jsonl", arm))
	done, err := resumeIDs(outPath)
	if err != nil {
		return err
	}
	fh, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer fh.Close()

	for _, it := range items {
		if done[it.ID] {
			continue
		}
		entry, ok := cache[it.ID]
		if !ok {
			return fmt.Errorf("no cached panel for %s", it.ID)
		}
		rec := produceArm(fusion, client, opt, it, entry, arm)
		if err := appendJSON(fh, rec); err != nil {
			return err
		}
	}
	fmt.Printf("arm %s -> %s\n", arm, outPath)
	return nil
}

// produceArm synthesizes a single item under one arm from its cached panel. A
// failed call is recorded as an error row (never aborts the run) so grading sees
// a strict paired set.
func produceArm(
	fusion *looper.FusionLooper,
	client *looper.Client,
	opt options,
	it item,
	entry cachedPanelItem,
	arm string,
) answerRecord {
	rec := answerRecord{ID: it.ID, Domain: it.Domain, Arm: arm, PanelSHA256: entry.PanelSHA256, Panel: []panelRec{}}

	if arm == "A" {
		req := buildRequest(it.Question, it.Context, opt)
		resp, err := client.CallModel(context.Background(), req, opt.judge, false, 1, nil, "")
		if err != nil {
			rec.Error = err.Error()
			return rec
		}
		rec.FinalAnswer = resp.Content
		rec.Usage = usageRec(resp.Usage)
		return rec
	}

	grounding, placebo := armGrounding(arm, opt.reference)
	if placebo {
		looper.SetGroundingBackends(placeboNLI(itemSeed(opt.placeboSeed, it.ID)), nil)
		defer looper.SetGroundingBackends(realNLI(), nil)
	}

	req := &looper.Request{
		OriginalRequest: buildRequest(it.Question, it.Context, opt),
		DecisionName:    "fusioneval",
		CachedPanel:     toModelResponses(entry.Panel),
		Algorithm: &config.AlgorithmConfig{
			Type: "fusion",
			Fusion: &config.FusionAlgorithmConfig{
				Model:          opt.judge,
				AnalysisModels: opt.panelModels,
				Grounding:      grounding,
			},
		},
	}
	resp, err := fusion.Execute(context.Background(), req)
	if err != nil {
		rec.Error = err.Error()
		return rec
	}
	rec.FinalAnswer = finalAnswer(resp.Body)
	rec.Usage = usageRec(resp.Usage)
	fillTrace(&rec, resp.IntermediateResponses)
	return rec
}

// armGrounding maps an arm label to its grounding config and whether the placebo
// NLI must be swapped in for this arm.
func armGrounding(arm, reference string) (cfg *config.FusionGroundingConfig, placebo bool) {
	switch arm {
	case "B":
		return nil, false // plain fusion
	case "C":
		return &config.FusionGroundingConfig{Enabled: true, Reference: reference}, false
	case "D":
		return &config.FusionGroundingConfig{Enabled: true, Reference: reference}, true
	case "annotate":
		return &config.FusionGroundingConfig{Enabled: true, Reference: reference, Policy: config.FusionGroundingPolicyAnnotate}, false
	case "filter":
		return &config.FusionGroundingConfig{Enabled: true, Reference: reference, Policy: config.FusionGroundingPolicyFilter, MinScore: 0.55, MinKeep: 1}, false
	default:
		// Treat unknown arms as plain fusion to avoid silent misconfiguration.
		return nil, false
	}
}

func fillTrace(rec *answerRecord, intermediate interface{}) {
	trace, ok := intermediate.(*looper.FusionTrace)
	if !ok || trace == nil || trace.Grounding == nil {
		return
	}
	rec.GroundingPresent = true
	rec.ReferenceMode = trace.Grounding.ReferenceMode
	rec.Policy = trace.Grounding.Policy
	rec.Panel = rec.Panel[:0]
	for _, s := range trace.Grounding.Scores {
		score := s.Score
		flagged := s.FlaggedSpans
		if flagged == nil {
			flagged = []string{}
		}
		rec.Panel = append(rec.Panel, panelRec{
			Model:          s.Model,
			GroundingScore: &score,
			Dropped:        s.Dropped,
			Flagged:        flagged,
		})
	}
}

// ---- helpers ----

func buildRequest(question, context string, opt options) *openai.ChatCompletionNewParams {
	msgs := []openai.ChatCompletionMessageParamUnion{}
	if strings.TrimSpace(context) != "" {
		msgs = append(msgs, openai.SystemMessage(context))
	}
	msgs = append(msgs, openai.UserMessage(question))
	req := &openai.ChatCompletionNewParams{Messages: msgs}
	if opt.temperature > 0 {
		req.Temperature = openai.Float(opt.temperature)
	}
	if opt.maxTokens > 0 {
		req.MaxCompletionTokens = openai.Int(int64(opt.maxTokens))
	}
	return req
}

func toModelResponses(panel []cachedResponse) []*looper.ModelResponse {
	out := make([]*looper.ModelResponse, 0, len(panel))
	for _, p := range panel {
		out = append(out, &looper.ModelResponse{
			Model:            p.Model,
			Content:          p.Content,
			ReasoningContent: p.Reasoning,
			Usage:            looper.TokenUsage(p.Usage),
		})
	}
	return out
}

func finalAnswer(body []byte) string {
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &completion); err != nil || len(completion.Choices) == 0 {
		return ""
	}
	return completion.Choices[0].Message.Content
}

func panelSHA256(panel []cachedResponse) string {
	data, _ := json.Marshal(panel)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func itemSeed(base uint64, id string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))
	return base ^ h.Sum64()
}

func loadItems(path string) ([]item, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []item
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var it item
		if err := json.Unmarshal([]byte(line), &it); err != nil {
			return nil, fmt.Errorf("bad item line: %w", err)
		}
		items = append(items, it)
	}
	return items, sc.Err()
}

func loadPanelCache(path string) (map[string]cachedPanelItem, error) {
	cache := map[string]cachedPanelItem{}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cache, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var e cachedPanelItem
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			return nil, err
		}
		cache[e.ItemID] = e
	}
	return cache, sc.Err()
}

func resumeIDs(path string) (map[string]bool, error) {
	done := map[string]bool{}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return done, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err == nil && rec.ID != "" {
			done[rec.ID] = true
		}
	}
	return done, sc.Err()
}

func appendJSON(fh *os.File, v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fh.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

var _ = time.Now // reserved for future per-arm timing
