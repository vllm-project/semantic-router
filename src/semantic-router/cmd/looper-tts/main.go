// Command looper-tts executes a saved fixed-budget Looper benchmark manifest.
//
// The Python harness is the portable entry point and provides the deterministic
// fake provider. This command is the native path for supported builds: it uses
// the production Looper implementations, attaches the same CallObserver seam,
// and emits the v1 records contract plus a runtime receipt.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

type manifest struct {
	SchemaVersion string         `json:"schema_version"`
	ExperimentID  string         `json:"experiment_id"`
	ConfigSHA256  string         `json:"config_sha256"`
	Config        manifestConfig `json:"config"`
	Matrix        []manifestCell `json:"matrix"`
}

type manifestConfig struct {
	Dataset manifestDataset  `json:"dataset"`
	Models  []manifestModel  `json:"models"`
	Arms    []manifestArm    `json:"arms"`
	Budgets []manifestBudget `json:"budgets"`
	Scorer  manifestScorer   `json:"scorer"`
}

type manifestDataset struct {
	EvidenceKind string         `json:"evidence_kind"`
	Items        []manifestItem `json:"items"`
}

type manifestItem struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
}

type manifestModel struct {
	ID       string           `json:"id"`
	Model    string           `json:"model"`
	Sampling manifestSampling `json:"sampling"`
	Pricing  manifestPricing  `json:"pricing"`
}

type manifestSampling struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

type manifestPricing struct {
	InputPerMillion  *float64 `json:"input_per_million"`
	OutputPerMillion *float64 `json:"output_per_million"`
}

type manifestArm struct {
	ID         string                 `json:"id"`
	Algorithm  string                 `json:"algorithm"`
	ModelIDs   []string               `json:"model_ids"`
	Parameters map[string]interface{} `json:"parameters"`
}

type manifestBudget struct {
	ID               string `json:"id"`
	MaxCalls         int64  `json:"max_calls"`
	MaxTotalTokens   int64  `json:"max_total_tokens"`
	ExhaustionPolicy string `json:"exhaustion_policy"`
}

type manifestScorer struct {
	ID string `json:"id"`
}

type manifestCell struct {
	ID       string   `json:"id"`
	ArmID    string   `json:"arm_id"`
	BudgetID string   `json:"budget_id"`
	Seed     int      `json:"seed"`
	ItemIDs  []string `json:"item_ids"`
}

type usageRecord struct {
	PromptTokens     *int64 `json:"prompt_tokens"`
	CompletionTokens *int64 `json:"completion_tokens"`
	TotalTokens      *int64 `json:"total_tokens"`
}

type callRecord struct {
	ID            string      `json:"id"`
	ExperimentID  string      `json:"experiment_id"`
	CellID        string      `json:"cell_id"`
	ItemID        string      `json:"item_id"`
	Stage         string      `json:"stage"`
	ModelID       string      `json:"model_id"`
	Attempt       int         `json:"attempt"`
	Status        string      `json:"status"`
	Usage         usageRecord `json:"usage"`
	LatencyMs     *int64      `json:"latency_ms"`
	RawOutputPath *string     `json:"raw_output_path"`
	Error         *string     `json:"error"`
	CacheID       *string     `json:"cache_id"`
}

type candidateScore struct {
	CallID string   `json:"call_id"`
	Score  *float64 `json:"score"`
}

type resultRecord struct {
	ID              string           `json:"id"`
	ExperimentID    string           `json:"experiment_id"`
	CellID          string           `json:"cell_id"`
	ItemID          string           `json:"item_id"`
	Status          string           `json:"status"`
	FinalAnswer     *string          `json:"final_answer"`
	Score           *float64         `json:"score"`
	ScorerID        string           `json:"scorer_id"`
	CallIDs         []string         `json:"call_ids"`
	CandidateScores []candidateScore `json:"candidate_scores"`
	PanelSHA256     *string          `json:"panel_sha256"`
	BudgetStatus    string           `json:"budget_status"`
	Error           *string          `json:"error"`
}

type recordsBundle struct {
	SchemaVersion string         `json:"schema_version"`
	ExperimentID  string         `json:"experiment_id"`
	EvidenceKind  string         `json:"evidence_kind"`
	Calls         []callRecord   `json:"calls"`
	Results       []resultRecord `json:"results"`
}

type options struct {
	Manifest       string
	Output         string
	Endpoint       string
	APIKeyEnv      string
	TimeoutSeconds int
}

func main() {
	opt := parseFlags()
	if err := run(opt); err != nil {
		fmt.Fprintln(os.Stderr, "looper-tts:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	opt := options{}
	flag.StringVar(&opt.Manifest, "manifest", "", "saved manifest.json")
	flag.StringVar(&opt.Output, "output", "", "output directory for records and receipt")
	flag.StringVar(&opt.Endpoint, "endpoint", "", "OpenAI-compatible chat-completions endpoint")
	flag.StringVar(&opt.APIKeyEnv, "api-key-env", "VLLM_API_KEY", "environment variable containing the provider key")
	flag.IntVar(&opt.TimeoutSeconds, "timeout", 600, "per-request timeout in seconds")
	flag.Parse()
	return opt
}

func run(opt options) error {
	if opt.Manifest == "" || opt.Output == "" || opt.Endpoint == "" {
		return fmt.Errorf("--manifest, --output and --endpoint are required")
	}
	data, err := os.ReadFile(opt.Manifest)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	var plan manifest
	if err := json.Unmarshal(data, &plan); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	if plan.SchemaVersion != "looper-tts.v1" || plan.ExperimentID == "" || plan.ConfigSHA256 == "" || plan.Config.Dataset.EvidenceKind == "" {
		return fmt.Errorf("manifest is missing schema, experiment identity, config digest or dataset evidence kind")
	}
	if opt.TimeoutSeconds <= 0 {
		return fmt.Errorf("--timeout must be positive")
	}
	if err := os.MkdirAll(opt.Output, 0o755); err != nil {
		return err
	}

	modelByID := make(map[string]manifestModel, len(plan.Config.Models))
	modelIDBySlug := make(map[string]string, len(plan.Config.Models))
	for _, model := range plan.Config.Models {
		modelByID[model.ID] = model
		// Preserve the manifest's declaration order when two local IDs point at
		// the same provider slug. The v1 evidence contract stores local IDs,
		// while the production client only sees the provider slug.
		if _, exists := modelIDBySlug[model.Model]; !exists {
			modelIDBySlug[model.Model] = model.ID
		}
	}
	armByID := make(map[string]manifestArm, len(plan.Config.Arms))
	for _, arm := range plan.Config.Arms {
		armByID[arm.ID] = arm
	}
	budgetByID := make(map[string]manifestBudget, len(plan.Config.Budgets))
	for _, budget := range plan.Config.Budgets {
		budgetByID[budget.ID] = budget
	}
	itemByID := make(map[string]manifestItem, len(plan.Config.Dataset.Items))
	for _, item := range plan.Config.Dataset.Items {
		itemByID[item.ID] = item
	}

	apiKey := ""
	if opt.APIKeyEnv != "" {
		apiKey = os.Getenv(opt.APIKeyEnv)
	}
	clientCfg := &config.LooperConfig{
		Endpoint:       opt.Endpoint,
		TimeoutSeconds: opt.TimeoutSeconds,
		Headers:        map[string]string{},
	}
	if apiKey != "" {
		clientCfg.Headers["Authorization"] = "Bearer " + apiKey
	}
	client, err := looper.NewConnectorClient(clientCfg)
	if err != nil {
		return fmt.Errorf("create Looper client: %w", err)
	}
	defer func() { _ = client.Close() }()

	bundle := recordsBundle{
		SchemaVersion: "looper-tts.v1",
		ExperimentID:  plan.ExperimentID,
		EvidenceKind:  plan.Config.Dataset.EvidenceKind,
		Calls:         []callRecord{},
		Results:       []resultRecord{},
	}
	receipt := map[string]interface{}{
		"schema_version": "looper-tts-runtime.v1",
		"experiment_id":  plan.ExperimentID,
		"config_sha256":  plan.ConfigSHA256,
		"execution": map[string]interface{}{
			"provider":        "native-looper",
			"timeout_seconds": opt.TimeoutSeconds,
		},
		"cells": []interface{}{},
	}

	for _, cell := range plan.Matrix {
		arm, ok := armByID[cell.ArmID]
		if !ok {
			return fmt.Errorf("matrix cell %q references unknown arm %q", cell.ID, cell.ArmID)
		}
		budget, ok := budgetByID[cell.BudgetID]
		if !ok {
			return fmt.Errorf("matrix cell %q references unknown budget %q", cell.ID, cell.BudgetID)
		}
		for _, itemID := range cell.ItemIDs {
			item, ok := itemByID[itemID]
			if !ok {
				return fmt.Errorf("cell %q references unknown item %q", cell.ID, itemID)
			}
			result, calls, itemReceipt := executeItem(
				client, clientCfg, plan, cell, arm, budget, item,
				modelByID, opt.Output,
				modelIDBySlug,
			)
			bundle.Results = append(bundle.Results, result)
			bundle.Calls = append(bundle.Calls, calls...)
			receipt["cells"] = append(receipt["cells"].([]interface{}), itemReceipt)
		}
	}

	if err := writeJSON(filepath.Join(opt.Output, "records.json"), bundle); err != nil {
		return err
	}
	return writeJSON(filepath.Join(opt.Output, "runtime_receipt.json"), receipt)
}

type itemExecutionReceipt struct {
	CellID string                   `json:"cell_id"`
	ItemID string                   `json:"item_id"`
	Status string                   `json:"status"`
	Budget looper.BudgetSnapshot    `json:"budget"`
	Calls  []map[string]interface{} `json:"calls"`
}

func executeItem(
	client *looper.Client,
	clientCfg *config.LooperConfig,
	plan manifest,
	cell manifestCell,
	arm manifestArm,
	budget manifestBudget,
	item manifestItem,
	modelByID map[string]manifestModel,
	outputDir string,
	modelIDBySlug map[string]string,
) (resultRecord, []callRecord, itemExecutionReceipt) {
	controller := looper.NewBudgetController(looper.BudgetLimits{
		MaxCalls: budget.MaxCalls, MaxTotalTokens: budget.MaxTotalTokens,
	})
	observer := &evidenceObserver{
		budget:        controller,
		experimentID:  plan.ExperimentID,
		cellID:        cell.ID,
		itemID:        item.ID,
		outputDir:     outputDir,
		arm:           arm,
		modelByID:     modelByID,
		modelIDBySlug: modelIDBySlug,
		pending:       map[string]pendingCall{},
		attempts:      map[string]int{},
	}
	ctx := looper.WithCallObserver(context.Background(), observer)

	resp, execErr := executeAlgorithm(ctx, client, clientCfg, arm, budget, item.Prompt, cell.Seed, modelByID)
	snapshot := controller.Snapshot()
	answer := ""
	if resp != nil {
		answer = responseAnswer(resp.Body)
	}
	status := "success"
	var resultErr *string
	if snapshot.Exhausted || looper.IsBudgetExhausted(execErr) {
		status = "budget_exhausted"
		message := "budget exhausted"
		if execErr != nil {
			message = execErr.Error()
		}
		resultErr = &message
	} else if execErr != nil || strings.TrimSpace(answer) == "" {
		status = "error"
		message := "algorithm produced no answer"
		if execErr != nil {
			message = execErr.Error()
		}
		resultErr = &message
	}
	var finalAnswer *string
	if answer != "" {
		finalAnswer = &answer
	}
	callIDs := make([]string, 0, len(observer.calls))
	for _, call := range observer.calls {
		callIDs = append(callIDs, call.ID)
	}
	result := resultRecord{
		ID:              stableID(plan.ExperimentID, cell.ID, item.ID, "result"),
		ExperimentID:    plan.ExperimentID,
		CellID:          cell.ID,
		ItemID:          item.ID,
		Status:          status,
		FinalAnswer:     finalAnswer,
		Score:           nil,
		ScorerID:        plan.Config.Scorer.ID,
		CallIDs:         callIDs,
		CandidateScores: []candidateScore{},
		PanelSHA256:     nil,
		BudgetStatus:    map[bool]string{true: "exhausted", false: "within"}[status == "budget_exhausted"],
		Error:           resultErr,
	}
	return result, observer.calls, itemExecutionReceipt{
		CellID: cell.ID, ItemID: item.ID, Status: status,
		Budget: snapshot, Calls: observer.receiptCalls,
	}
}

func executeAlgorithm(
	ctx context.Context,
	client *looper.Client,
	clientCfg *config.LooperConfig,
	arm manifestArm,
	budget manifestBudget,
	prompt string,
	seed int,
	modelByID map[string]manifestModel,
) (*looper.Response, error) {
	sampling := make(map[string]looper.ModelSampling, len(arm.ModelIDs))
	for _, id := range arm.ModelIDs {
		model, ok := modelByID[id]
		if !ok {
			return nil, fmt.Errorf("arm %q references unknown model %q", arm.ID, id)
		}
		declared := looper.ModelSampling{Temperature: model.Sampling.Temperature, TopP: model.Sampling.TopP}
		if previous, exists := sampling[model.Model]; exists && previous != declared {
			return nil, fmt.Errorf("native runner cannot distinguish different sampling settings for provider model %q", model.Model)
		}
		sampling[model.Model] = declared
	}
	ctx = looper.WithModelSampling(ctx, sampling)
	maxTokens := budget.MaxTotalTokens / budget.MaxCalls
	if maxTokens < 1 {
		maxTokens = 1
	}
	if maxTokens > 1024 {
		maxTokens = 1024
	}
	refs := make([]config.ModelRef, 0, len(arm.ModelIDs))
	params := make(map[string]config.ModelParams, len(arm.ModelIDs))
	for _, modelID := range arm.ModelIDs {
		model, ok := modelByID[modelID]
		if !ok {
			return nil, fmt.Errorf("arm %q references unknown model %q", arm.ID, modelID)
		}
		refs = append(refs, config.ModelRef{Model: model.Model})
		pricing := config.ModelPricing{}
		if model.Pricing.InputPerMillion != nil {
			pricing.PromptPer1M = *model.Pricing.InputPerMillion
		}
		if model.Pricing.OutputPerMillion != nil {
			pricing.CompletionPer1M = *model.Pricing.OutputPerMillion
		}
		pricing.Currency = "USD"
		params[model.Model] = config.ModelParams{Pricing: pricing}
	}
	request := &openai.ChatCompletionNewParams{
		Messages:            []openai.ChatCompletionMessageParamUnion{openai.UserMessage(prompt)},
		MaxCompletionTokens: openai.Int(maxTokens),
		Seed:                openai.Int(int64(seed)),
	}
	if len(arm.ModelIDs) > 0 {
		model := modelByID[arm.ModelIDs[0]]
		request.Model = openai.ChatModel(model.Model)
		request.Temperature = openai.Float(model.Sampling.Temperature)
		request.TopP = openai.Float(model.Sampling.TopP)
	}
	req := &looper.Request{
		OriginalRequest: request,
		ModelRefs:       refs,
		ModelParams:     params,
		DecisionName:    "looper-tts-" + arm.ID,
		IsStreaming:     false,
	}
	if arm.Algorithm == "direct" {
		if len(arm.ModelIDs) != 1 {
			return nil, fmt.Errorf("direct arm %q must have one model", arm.ID)
		}
		modelResponse, err := client.CallModelWithOptions(ctx, *request, looper.ModelTarget{Name: refs[0].Model}, looper.CallOptions{
			DecisionName: req.DecisionName, Iteration: 1, Mode: looper.ResponseJSON,
			Stage: looper.CallStageGenerate, Role: "candidate",
		})
		if err != nil {
			return nil, err
		}
		if modelResponse == nil {
			return nil, fmt.Errorf("direct model call returned no response")
		}
		return &looper.Response{
			Body:          modelResponse.Raw,
			ContentType:   "application/json",
			Model:         modelResponse.Model,
			ModelsUsed:    []string{modelResponse.Model},
			Iterations:    1,
			AlgorithmType: "direct",
			Usage:         modelResponse.Usage,
			LatencyMs:     modelResponse.LatencyMs,
		}, nil
	}
	var algorithm *config.AlgorithmConfig
	switch arm.Algorithm {
	case "confidence":
		threshold, _ := arm.Parameters["threshold"].(float64)
		algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmConfidence, Confidence: &config.ConfidenceAlgorithmConfig{
			ConfidenceMethod: "self_verify", Threshold: threshold, EscalationOrder: config.ConfidenceEscalationOrderDeclared,
		}}
	case "remom":
		breadth := intParameters(arm.Parameters["breadth"])
		algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{
			BreadthSchedule: breadth, SynthesisModel: refs[0].Model, MaxCompletionTokens: intPointer(maxTokens),
			ModelDistribution: config.ReMoMDistributionRoundRobin, ShuffleSeed: seed,
		}}
	case "fusion":
		panel := stringParameters(arm.Parameters["panel_model_ids"])
		judgeID, _ := arm.Parameters["judge_model_id"].(string)
		synthesisID, _ := arm.Parameters["synthesis_model_id"].(string)
		judge := modelByID[judgeID]
		synthesis := modelByID[synthesisID]
		if judge.Model != synthesis.Model {
			return nil, fmt.Errorf("native Fusion runner requires judge_model_id and synthesis_model_id to resolve to the same model")
		}
		panelSlugs := make([]string, 0, len(panel))
		for _, id := range panel {
			panelSlugs = append(panelSlugs, modelByID[id].Model)
		}
		algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
			Model: judge.Model, AnalysisModels: panelSlugs, AnalysisMode: config.FusionAnalysisModeSeparate,
			MaxCompletionTokens: int(maxTokens), MinSuccessfulResponses: len(panelSlugs),
		}}
	default:
		return nil, fmt.Errorf("unsupported benchmark algorithm %q", arm.Algorithm)
	}
	req.Algorithm = algorithm
	_ = clientCfg // retained in the signature to make endpoint ownership explicit at the call site
	native, err := looper.FactoryWithClient(clientCfg, arm.Algorithm, client)
	if err != nil {
		return nil, err
	}
	return looper.ExecuteWithLatency(ctx, native, req)
}

type pendingCall struct {
	ID      string
	Info    looper.CallInfo
	Attempt int
}

type evidenceObserver struct {
	mu            sync.Mutex
	budget        *looper.BudgetController
	experimentID  string
	cellID        string
	itemID        string
	outputDir     string
	arm           manifestArm
	modelByID     map[string]manifestModel
	modelIDBySlug map[string]string
	pending       map[string]pendingCall
	attempts      map[string]int
	ordinal       int
	calls         []callRecord
	receiptCalls  []map[string]interface{}
}

func (o *evidenceObserver) BeforeCall(ctx context.Context, info looper.CallInfo) (*looper.CallReservation, error) {
	reservation, err := o.budget.BeforeCall(ctx, info)
	if err != nil {
		return nil, err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.ordinal++
	key := info.Stage + "\x00" + info.Model
	o.attempts[key]++
	attempt := o.attempts[key]
	id := stableID(o.experimentID, o.cellID, o.itemID, info.Stage, info.Model, fmt.Sprintf("%d", attempt), fmt.Sprintf("%d", o.ordinal))
	o.pending[reservation.ID] = pendingCall{ID: id, Info: info, Attempt: attempt}
	return reservation, nil
}

func (o *evidenceObserver) AfterCall(ctx context.Context, info looper.CallInfo, reservation *looper.CallReservation, result looper.CallResult) {
	o.budget.AfterCall(ctx, info, reservation, result)
	if reservation == nil {
		return
	}
	o.mu.Lock()
	pending, ok := o.pending[reservation.ID]
	delete(o.pending, reservation.ID)
	o.mu.Unlock()
	if !ok {
		return
	}
	call := callRecord{
		ID: pending.ID, ExperimentID: o.experimentID, CellID: o.cellID, ItemID: o.itemID,
		Stage: info.Stage, ModelID: o.localModelID(info), Attempt: pending.Attempt,
		Status: "success", Usage: usageRecordFor(result.Response), CacheID: nil,
	}
	if result.Err != nil {
		message := truncate(result.Err.Error())
		call.Status = "error"
		call.Error = &message
	}
	if result.Response != nil {
		latency := result.Response.LatencyMs
		if result.LatencyMs >= 0 {
			latency = result.LatencyMs
		}
		call.LatencyMs = &latency
		if result.Err == nil && len(result.Response.Raw) > 0 {
			path, err := o.writeRaw(pending.ID, result.Response.Raw)
			if err == nil {
				call.RawOutputPath = &path
			} else {
				message := truncate(err.Error())
				call.Status = "error"
				call.Error = &message
			}
		}
	}
	o.mu.Lock()
	o.calls = append(o.calls, call)
	model := o.modelByID[o.localModelID(info)]
	usage := call.Usage
	event := map[string]interface{}{
		"call_id": call.ID, "stage": call.Stage, "model_id": call.ModelID,
		"attempt": call.Attempt, "status": call.Status,
		"estimated_tokens": info.EstimatedTotalTokens,
		"usage_source":     usageSource(result.Response),
		"usage_known":      result.Response != nil && result.Response.UsageKnown(),
		"latency_ms":       call.LatencyMs,
		"cost_usd":         usageCost(usage, model),
	}
	o.receiptCalls = append(o.receiptCalls, event)
	o.mu.Unlock()
}

func (o *evidenceObserver) localModelID(info looper.CallInfo) string {
	// The production client sends provider slugs in CallInfo.Model, while the
	// evidence contract uses local manifest IDs. Prefer the stage declaration so
	// two local IDs may safely refer to the same provider slug (a common way to
	// compare sampling or revision variants).
	candidates := o.arm.ModelIDs
	if o.arm.Algorithm == "fusion" {
		switch info.Stage {
		case looper.CallStageJudge:
			if id, ok := o.arm.Parameters["judge_model_id"].(string); ok {
				candidates = []string{id}
			}
		case looper.CallStageSynthesize:
			if id, ok := o.arm.Parameters["synthesis_model_id"].(string); ok {
				candidates = []string{id}
			}
		case looper.CallStageGenerate:
			candidates = stringParameters(o.arm.Parameters["panel_model_ids"])
		}
	}
	for _, id := range candidates {
		if model, ok := o.modelByID[id]; ok && model.Model == info.Model {
			return id
		}
	}
	if id, ok := o.modelIDBySlug[info.Model]; ok {
		return id
	}
	return info.Model
}

func (o *evidenceObserver) writeRaw(callID string, body []byte) (string, error) {
	// Manifest IDs are data and may contain path separators. Hashing directory
	// components keeps artifacts below the selected output root.
	relative := filepath.ToSlash(filepath.Join("raw", stableID(o.cellID), stableID(o.itemID), callID+".json"))
	path := filepath.Join(o.outputDir, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, body, 0o644); err != nil {
		return "", err
	}
	return relative, nil
}

func usageRecordFor(response *looper.ModelResponse) usageRecord {
	if response == nil {
		return usageRecord{}
	}
	usage := response.Usage
	record := usageRecord{}
	if response.UsagePresent.PromptTokens {
		value := usage.PromptTokens
		record.PromptTokens = &value
	}
	if response.UsagePresent.CompletionTokens {
		value := usage.CompletionTokens
		record.CompletionTokens = &value
	}
	if response.UsagePresent.TotalTokens {
		value := usage.TotalTokens
		record.TotalTokens = &value
	}
	return record
}

func usageSource(response *looper.ModelResponse) string {
	if response != nil && response.UsageKnown() {
		return "provider"
	}
	return "reservation"
}

func usageCost(usage usageRecord, model manifestModel) interface{} {
	if usage.PromptTokens == nil || usage.CompletionTokens == nil || model.Pricing.InputPerMillion == nil || model.Pricing.OutputPerMillion == nil {
		return nil
	}
	return (float64(*usage.PromptTokens)*(*model.Pricing.InputPerMillion) + float64(*usage.CompletionTokens)*(*model.Pricing.OutputPerMillion)) / 1_000_000.0
}

func responseAnswer(body []byte) string {
	var payload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || len(payload.Choices) == 0 {
		return ""
	}
	return payload.Choices[0].Message.Content
}

func intParameters(value interface{}) []int {
	values, _ := value.([]interface{})
	result := make([]int, 0, len(values))
	for _, value := range values {
		if number, ok := value.(float64); ok {
			result = append(result, int(number))
		}
	}
	return result
}

func stringParameters(value interface{}) []string {
	values, _ := value.([]interface{})
	result := make([]string, 0, len(values))
	for _, value := range values {
		if entry, ok := value.(string); ok {
			result = append(result, entry)
		}
	}
	return result
}

func intPointer(value int64) *int {
	converted := int(value)
	return &converted
}

func stableID(parts ...string) string {
	hash := sha256.New()
	for _, part := range parts {
		_, _ = hash.Write([]byte(part))
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func truncate(value string) string {
	if len(value) > 1000 {
		return value[:1000]
	}
	return value
}

func writeJSON(path string, value interface{}) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
