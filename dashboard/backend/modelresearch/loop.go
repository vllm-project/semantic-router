package modelresearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type trialPlan struct {
	Name    string
	Params  map[string]any
	UseLoRA bool
}

func (m *Manager) runTraining(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	campaign *Campaign,
	plan trialPlan,
	trialDir string,
	device string,
	trialIndex int,
) (string, map[string]string, error) {
	switch def.Key {
	case "feedback":
		return m.runFeedbackTraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	case "fact-check":
		return m.runFactCheckTraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	case "jailbreak":
		return m.runJailbreakTraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	case "intent":
		return m.runIntentTraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	case "pii":
		return m.runPIITraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	case "domain":
		return m.runDomainTraining(ctx, campaignID, campaign, plan, trialDir, device, trialIndex)
	default:
		return "", nil, fmt.Errorf("unsupported recipe %q", def.Key)
	}
}

func (m *Manager) runFeedbackTraining(ctx context.Context, campaignID string, campaign *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "user_feedback_classifier")
	outputBase := filepath.Join(trialDir, "feedback-detector")
	args := []string{
		filepath.Join(scriptDir, "train_feedback_detector.py"),
		"--model_name", stringHint(plan.Params, "model_name", "llm-semantic-router/mmbert-32k-yarn"),
		"--output_dir", outputBase,
		"--epochs", strconv.Itoa(intHint(plan.Params, "epochs", 10)),
		"--batch_size", strconv.Itoa(intHint(plan.Params, "batch_size", 16)),
		"--lr", fmt.Sprintf("%.6f", floatHint(plan.Params, "learning_rate", 2e-5)),
		"--lora_rank", strconv.Itoa(intHint(plan.Params, "lora_rank", 64)),
		"--lora_alpha", strconv.Itoa(intHint(plan.Params, "lora_alpha", 128)),
		"--use_lora",
	}
	if dataSource := strings.TrimSpace(campaign.Overrides.DatasetOverride); dataSource != "" {
		args = append(args, "--data_source", dataSource)
	}
	env := trainingEnv(device)
	if err := m.executePython(ctx, commandSpec{Dir: scriptDir, Name: m.pythonPath, Args: args, Env: env}, campaign.ID, trialIndex); err != nil {
		return "", nil, err
	}
	return outputBase + "_lora", map[string]string{"training_dir": outputBase + "_lora"}, nil
}

func (m *Manager) runFactCheckTraining(ctx context.Context, campaignID string, campaign *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "fact_check_fine_tuning_lora")
	outputDir := filepath.Join(trialDir, "fact-check-lora")
	args := []string{
		filepath.Join(scriptDir, "fact_check_bert_finetuning_lora.py"),
		"--mode", "train",
		"--model", stringHint(plan.Params, "model", "mmbert-32k"),
		"--output-dir", outputDir,
		"--epochs", strconv.Itoa(intHint(plan.Params, "epochs", 5)),
		"--batch-size", strconv.Itoa(intHint(plan.Params, "batch_size", 16)),
		"--learning-rate", fmt.Sprintf("%.6f", floatHint(plan.Params, "learning_rate", 3e-5)),
		"--lora-rank", strconv.Itoa(intHint(plan.Params, "lora_rank", 16)),
		"--lora-alpha", strconv.Itoa(intHint(plan.Params, "lora_alpha", 32)),
	}
	if dataDir := datasetDirOverride(campaign.Overrides.DatasetOverride); dataDir != "" {
		args = append(args, "--data-dir", dataDir)
	}
	if err := m.executePython(ctx, commandSpec{Dir: scriptDir, Name: m.pythonPath, Args: args, Env: trainingEnv(device)}, campaign.ID, trialIndex); err != nil {
		return "", nil, err
	}
	return outputDir, map[string]string{"training_dir": outputDir}, nil
}

func (m *Manager) runJailbreakTraining(ctx context.Context, campaignID string, _ *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "prompt_guard_fine_tuning_lora")
	outputDir := filepath.Join(trialDir, "jailbreak-lora")
	args := []string{
		filepath.Join(scriptDir, "jailbreak_bert_finetuning_lora.py"),
		"--mode", "train",
		"--model", stringHint(plan.Params, "model", "mmbert-32k"),
		"--output-dir", outputDir,
		"--epochs", strconv.Itoa(intHint(plan.Params, "epochs", 3)),
		"--batch-size", strconv.Itoa(intHint(plan.Params, "batch_size", 8)),
		"--learning-rate", fmt.Sprintf("%.6f", floatHint(plan.Params, "learning_rate", 3e-5)),
		"--lora-rank", strconv.Itoa(intHint(plan.Params, "lora_rank", 8)),
		"--lora-alpha", strconv.Itoa(intHint(plan.Params, "lora_alpha", 16)),
	}
	if err := m.executePython(ctx, commandSpec{Dir: scriptDir, Name: m.pythonPath, Args: args, Env: trainingEnv(device)}, campaignID, trialIndex); err != nil {
		return "", nil, err
	}
	return outputDir, map[string]string{"training_dir": outputDir}, nil
}

func (m *Manager) runIntentTraining(ctx context.Context, campaignID string, _ *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	return m.runIntentStyleTraining(ctx, campaignID, plan, filepath.Join(trialDir, "intent-lora"), device, trialIndex)
}

func (m *Manager) runDomainTraining(ctx context.Context, campaignID string, _ *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	return m.runIntentStyleTraining(ctx, campaignID, plan, filepath.Join(trialDir, "domain-lora"), device, trialIndex)
}

func (m *Manager) runIntentStyleTraining(ctx context.Context, campaignID string, plan trialPlan, outputDir string, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "classifier_model_fine_tuning_lora")
	args := []string{
		filepath.Join(scriptDir, "ft_linear_lora.py"),
		"--mode", "train",
		"--model", stringHint(plan.Params, "model", "mmbert-32k"),
		"--output-dir", outputDir,
		"--epochs", strconv.Itoa(intHint(plan.Params, "epochs", 3)),
		"--batch-size", strconv.Itoa(intHint(plan.Params, "batch_size", 8)),
		"--learning-rate", fmt.Sprintf("%.6f", floatHint(plan.Params, "learning_rate", 3e-5)),
		"--lora-rank", strconv.Itoa(intHint(plan.Params, "lora_rank", 32)),
		"--lora-alpha", strconv.Itoa(intHint(plan.Params, "lora_alpha", 64)),
		"--max-samples", strconv.Itoa(intHint(plan.Params, "max_samples", 5000)),
	}
	if boolHint(plan.Params, "enable_feature_alignment", false) {
		args = append(args, "--enable-feature-alignment")
		args = append(args, "--alignment-weight", fmt.Sprintf("%.4f", floatHint(plan.Params, "alignment_weight", 0.1)))
	}
	if err := m.executePython(ctx, commandSpec{Dir: scriptDir, Name: m.pythonPath, Args: args, Env: trainingEnv(device)}, campaignID, trialIndex); err != nil {
		return "", nil, err
	}
	return outputDir, map[string]string{"training_dir": outputDir}, nil
}

func (m *Manager) runPIITraining(ctx context.Context, campaignID string, _ *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "pii_model_fine_tuning_lora")
	modelName := stringHint(plan.Params, "model", "mmbert-32k")
	loraRank := intHint(plan.Params, "lora_rank", 8)
	outputDir := filepath.Join(trialDir, fmt.Sprintf("lora_pii_detector_%s_r%d_token_model", modelName, loraRank))
	args := []string{
		filepath.Join(scriptDir, "pii_bert_finetuning_lora.py"),
		"--mode", "train",
		"--model", modelName,
		"--epochs", strconv.Itoa(intHint(plan.Params, "epochs", 3)),
		"--batch-size", strconv.Itoa(intHint(plan.Params, "batch_size", 8)),
		"--learning-rate", fmt.Sprintf("%.6f", floatHint(plan.Params, "learning_rate", 1e-4)),
		"--lora-rank", strconv.Itoa(loraRank),
		"--lora-alpha", strconv.Itoa(intHint(plan.Params, "lora_alpha", 16)),
		"--max-samples", strconv.Itoa(intHint(plan.Params, "max_samples", 5000)),
	}
	if !boolHint(plan.Params, "use_ai4privacy", true) {
		args = append(args, "--no-ai4privacy")
	}
	if err := m.executePython(ctx, commandSpec{Dir: trialDir, Name: m.pythonPath, Args: args, Env: trainingEnv(device)}, campaignID, trialIndex); err != nil {
		return "", nil, err
	}
	return outputDir, map[string]string{"training_dir": outputDir}, nil
}

func (m *Manager) executePython(ctx context.Context, spec commandSpec, campaignID string, trialIndex int) error {
	return m.runCommand(ctx, spec, func(stream, line string) {
		line = strings.TrimSpace(line)
		if line == "" {
			return
		}
		m.recordEvent(campaignID, CampaignEvent{
			Timestamp:  time.Now().UTC(),
			Kind:       EventLog,
			Level:      strings.ToLower(stream),
			Message:    line,
			TrialIndex: trialIndex,
		})
	})
}

func (m *Manager) runOfflineEval(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	modelID string,
	useLoRA bool,
	datasetOverride string,
	outputDir string,
	device string,
	trialIndex int,
) (*MetricSnapshot, error) {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, err
	}
	modelKey := offlineEvalModelKey(def.Key)
	if modelKey == "" {
		return nil, fmt.Errorf("no offline eval mapping for %s", def.Key)
	}

	args := []string{
		filepath.Join(m.repoRoot, "src", "training", "model_eval", "mom_collection_eval.py"),
		"--model", modelKey,
		"--device", device,
		"--output_dir", outputDir,
	}
	if strings.TrimSpace(modelID) != "" {
		args = append(args, "--model_id", modelID)
	}
	if useLoRA {
		args = append(args, "--use_lora")
	}
	if customPath := offlineDatasetOverride(def.Key, datasetOverride); customPath != "" {
		args = append(args, "--custom_dataset", customPath)
	}

	if err := m.executePython(ctx, commandSpec{
		Dir:  m.repoRoot,
		Name: m.pythonPath,
		Args: args,
		Env:  trainingEnv(device),
	}, campaignID, trialIndex); err != nil {
		return nil, err
	}

	statsPath := filepath.Join(outputDir, fmt.Sprintf("%s_results.json", modelKey))
	if _, err := os.Stat(statsPath); err != nil {
		return nil, fmt.Errorf(
			"offline evaluation did not produce %s; inspect the preceding model_eval logs for device or model load errors",
			filepath.Base(statsPath),
		)
	}
	return readOfflineMetrics(statsPath, def.DefaultDataset, modelID)
}

func (m *Manager) runSignalEval(
	ctx context.Context,
	campaignID string,
	datasetID string,
	apiBase string,
	outputDir string,
	trialIndex int,
) (*MetricSnapshot, error) {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, err
	}
	outputPath := filepath.Join(outputDir, "signal-eval.json")
	args := []string{
		filepath.Join(m.repoRoot, "src", "training", "model_eval", "signal_eval.py"),
		"--dataset", datasetID,
		"--endpoint", strings.TrimRight(apiBase, "/") + "/v1/eval",
		"--output", outputPath,
		"--max_samples", "100",
	}
	if err := m.executePython(ctx, commandSpec{
		Dir:  m.repoRoot,
		Name: m.pythonPath,
		Args: args,
	}, campaignID, trialIndex); err != nil {
		return nil, err
	}
	return readSignalMetrics(outputPath, datasetID)
}

func (m *Manager) writeConfigFragment(campaignID string, baseline *MetricSnapshot, bestTrial *TrialResult) (string, error) {
	campaign := m.GetCampaign(campaignID)
	if campaign == nil || bestTrial == nil || bestTrial.Eval == nil {
		return "", errors.New("campaign or best trial unavailable")
	}
	payload := map[string]any{
		"candidate": map[string]any{
			"campaign_id":        campaign.ID,
			"name":               campaign.Name,
			"target":             campaign.Target,
			"goal_template":      campaign.GoalTemplate,
			"signal_hypothesis":  campaign.SignalHypothesis,
			"platform":           campaign.Platform,
			"model_path":         bestTrial.ModelPath,
			"use_lora":           bestTrial.UseLoRA,
			"primary_metric":     campaign.PrimaryMetric,
			"baseline_accuracy":  baseline.Accuracy,
			"candidate_accuracy": bestTrial.Eval.Accuracy,
			"improvement_pp":     bestTrial.Eval.ImprovementPP,
			"request_model":      campaign.RequestModel,
			"api_base":           campaign.APIBase,
			"default_dataset":    campaign.Recipe.DefaultDataset,
		},
	}
	path := filepath.Join(campaign.ArtifactDir, "candidate-config.yaml")
	data, err := yaml.Marshal(payload)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func readOfflineMetrics(path string, dataset string, modelID string) (*MetricSnapshot, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var stats struct {
		Accuracy  float64 `json:"accuracy"`
		F1        float64 `json:"f1"`
		Precision float64 `json:"precision"`
		Recall    float64 `json:"recall"`
		Latency   struct {
			AvgMS float64 `json:"avg_ms"`
		} `json:"latency"`
	}
	if err := json.Unmarshal(payload, &stats); err != nil {
		return nil, err
	}
	return &MetricSnapshot{
		Source:       "offline_eval",
		Dataset:      dataset,
		Accuracy:     stats.Accuracy,
		F1:           stats.F1,
		Precision:    stats.Precision,
		Recall:       stats.Recall,
		LatencyAvgMS: stats.Latency.AvgMS,
		OutputPath:   path,
		ModelID:      modelID,
	}, nil
}

func readSignalMetrics(path string, dataset string) (*MetricSnapshot, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var stats struct {
		Accuracy float64 `json:"accuracy"`
	}
	if err := json.Unmarshal(payload, &stats); err != nil {
		return nil, err
	}
	return &MetricSnapshot{
		Source:     "runtime_signal_eval",
		Dataset:    dataset,
		Accuracy:   stats.Accuracy,
		OutputPath: path,
	}, nil
}

func buildTrialPlans(def recipeDefinition, maxTrials int, hints map[string]any) ([]trialPlan, error) {
	allowed := allowedHintKeys(def.Key)
	for key := range hints {
		if _, ok := allowed[key]; !ok {
			return nil, fmt.Errorf("unsupported hyperparameter hint %q for %s", key, def.Key)
		}
	}

	plans := []trialPlan{
		{Name: "trial-01", UseLoRA: true, Params: defaultPlan(def.Key, 0)},
		{Name: "trial-02", UseLoRA: true, Params: defaultPlan(def.Key, 1)},
		{Name: "trial-03", UseLoRA: true, Params: defaultPlan(def.Key, 2)},
	}

	if maxTrials < len(plans) {
		plans = append([]trialPlan(nil), plans[:maxTrials]...)
	}
	for i := range plans {
		for key, value := range hints {
			plans[i].Params[key] = value
		}
	}
	return plans, nil
}

func defaultPlan(recipeKey string, variant int) map[string]any {
	candidates := map[string][]map[string]any{
		"feedback": {
			{"model_name": "llm-semantic-router/mmbert-32k-yarn", "epochs": 10, "batch_size": 16, "learning_rate": 2e-5, "lora_rank": 64, "lora_alpha": 128},
			{"model_name": "llm-semantic-router/mmbert-32k-yarn", "epochs": 8, "batch_size": 16, "learning_rate": 3e-5, "lora_rank": 32, "lora_alpha": 64},
			{"model_name": "llm-semantic-router/mmbert-32k-yarn", "epochs": 12, "batch_size": 8, "learning_rate": 2e-5, "lora_rank": 96, "lora_alpha": 192},
		},
		"fact-check": {
			{"model": "mmbert-32k", "epochs": 5, "batch_size": 16, "learning_rate": 3e-5, "lora_rank": 16, "lora_alpha": 32},
			{"model": "mmbert-base", "epochs": 6, "batch_size": 12, "learning_rate": 2e-5, "lora_rank": 16, "lora_alpha": 32},
			{"model": "modernbert-base", "epochs": 5, "batch_size": 12, "learning_rate": 2e-5, "lora_rank": 32, "lora_alpha": 64},
		},
		"jailbreak": {
			{"model": "mmbert-32k", "epochs": 3, "batch_size": 8, "learning_rate": 3e-5, "lora_rank": 8, "lora_alpha": 16},
			{"model": "mmbert-base", "epochs": 4, "batch_size": 8, "learning_rate": 2e-5, "lora_rank": 16, "lora_alpha": 32},
			{"model": "roberta-base", "epochs": 4, "batch_size": 12, "learning_rate": 2e-5, "lora_rank": 16, "lora_alpha": 32},
		},
		"intent": {
			{"model": "mmbert-32k", "epochs": 3, "batch_size": 8, "learning_rate": 3e-5, "lora_rank": 32, "lora_alpha": 64, "max_samples": 5000},
			{"model": "mmbert-base", "epochs": 4, "batch_size": 8, "learning_rate": 2e-5, "lora_rank": 32, "lora_alpha": 64, "enable_feature_alignment": true, "alignment_weight": 0.1, "max_samples": 5000},
			{"model": "modernbert-base", "epochs": 3, "batch_size": 8, "learning_rate": 3e-5, "lora_rank": 16, "lora_alpha": 32, "max_samples": 4000},
		},
		"pii": {
			{"model": "mmbert-32k", "epochs": 3, "batch_size": 8, "learning_rate": 1e-4, "lora_rank": 8, "lora_alpha": 16, "max_samples": 5000, "use_ai4privacy": true},
			{"model": "mmbert-base", "epochs": 4, "batch_size": 8, "learning_rate": 8e-5, "lora_rank": 16, "lora_alpha": 32, "max_samples": 6000, "use_ai4privacy": true},
			{"model": "modernbert-base", "epochs": 3, "batch_size": 8, "learning_rate": 7e-5, "lora_rank": 16, "lora_alpha": 32, "max_samples": 4000, "use_ai4privacy": false},
		},
		"domain": {
			{"model": "mmbert-32k", "epochs": 3, "batch_size": 8, "learning_rate": 3e-5, "lora_rank": 32, "lora_alpha": 64, "max_samples": 5000},
			{"model": "mmbert-base", "epochs": 4, "batch_size": 8, "learning_rate": 2e-5, "lora_rank": 32, "lora_alpha": 64, "enable_feature_alignment": true, "alignment_weight": 0.1, "max_samples": 5000},
			{"model": "modernbert-base", "epochs": 3, "batch_size": 8, "learning_rate": 3e-5, "lora_rank": 16, "lora_alpha": 32, "max_samples": 4000},
		},
	}
	variants := candidates[recipeKey]
	if len(variants) == 0 {
		return map[string]any{}
	}
	if variant >= len(variants) {
		variant = len(variants) - 1
	}
	return cloneMap(variants[variant])
}

func allowedHintKeys(recipeKey string) map[string]struct{} {
	common := map[string]struct{}{
		"epochs":        {},
		"batch_size":    {},
		"learning_rate": {},
		"lora_rank":     {},
		"lora_alpha":    {},
	}
	switch recipeKey {
	case "feedback":
		common["model_name"] = struct{}{}
	case "fact-check", "jailbreak", "intent", "pii", "domain":
		common["model"] = struct{}{}
	}
	if recipeKey == "intent" || recipeKey == "domain" {
		common["max_samples"] = struct{}{}
		common["enable_feature_alignment"] = struct{}{}
		common["alignment_weight"] = struct{}{}
	}
	if recipeKey == "pii" {
		common["max_samples"] = struct{}{}
		common["use_ai4privacy"] = struct{}{}
	}
	return common
}

func runtimeDatasetForRecipe(recipeKey string, override string) string {
	switch recipeKey {
	case "feedback":
		if strings.TrimSpace(override) != "" && !strings.Contains(override, "/") && !strings.Contains(override, ".") {
			return override
		}
		return "feedback-en"
	case "fact-check":
		if strings.TrimSpace(override) != "" && !strings.Contains(override, "/") && !strings.Contains(override, ".") {
			return override
		}
		return "fact-check-en"
	case "intent":
		if strings.TrimSpace(override) != "" && !strings.Contains(override, "/") && !strings.Contains(override, ".") {
			return override
		}
		return "mmlu-prox-en"
	case "domain":
		if strings.TrimSpace(override) != "" && !strings.Contains(override, "/") && !strings.Contains(override, ".") {
			return override
		}
		return "mmlu-prox-en"
	default:
		return ""
	}
}

func offlineEvalModelKey(recipeKey string) string {
	switch recipeKey {
	case "feedback":
		return "feedback"
	case "fact-check":
		return "fact-check"
	case "jailbreak":
		return "jailbreak"
	case "intent":
		return "intent"
	case "pii":
		return "pii"
	case "domain":
		return "intent"
	default:
		return ""
	}
}

func offlineDatasetOverride(recipeKey string, override string) string {
	if strings.TrimSpace(override) == "" {
		return ""
	}
	if (recipeKey == "intent" || recipeKey == "domain" || recipeKey == "pii") &&
		!strings.Contains(override, ".json") && !strings.Contains(override, ".csv") {
		return ""
	}
	return override
}

func datasetDirOverride(override string) string {
	if strings.TrimSpace(override) == "" {
		return ""
	}
	if info, err := os.Stat(override); err == nil && info.IsDir() {
		return override
	}
	return ""
}

func trainingEnv(device string) []string {
	if strings.EqualFold(device, "cuda") {
		return nil
	}
	return []string{
		"CUDA_VISIBLE_DEVICES=",
		"HIP_VISIBLE_DEVICES=",
		"ROCR_VISIBLE_DEVICES=",
	}
}

func cloneMap(input map[string]any) map[string]any {
	result := make(map[string]any, len(input))
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result[key] = input[key]
	}
	return result
}

func intHint(input map[string]any, key string, fallback int) int {
	value, ok := input[key]
	if !ok {
		return fallback
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		parsed, err := strconv.Atoi(strings.TrimSpace(typed))
		if err == nil {
			return parsed
		}
	}
	return fallback
}

func floatHint(input map[string]any, key string, fallback float64) float64 {
	value, ok := input[key]
	if !ok {
		return fallback
	}
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		if err == nil {
			return parsed
		}
	}
	return fallback
}

func stringHint(input map[string]any, key string, fallback string) string {
	value, ok := input[key]
	if !ok {
		return fallback
	}
	if typed, ok := value.(string); ok && strings.TrimSpace(typed) != "" {
		return typed
	}
	return fallback
}

func boolHint(input map[string]any, key string, fallback bool) bool {
	value, ok := input[key]
	if !ok {
		return fallback
	}
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		return strings.EqualFold(strings.TrimSpace(typed), "true")
	}
	return fallback
}
