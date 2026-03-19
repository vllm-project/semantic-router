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

func (m *Manager) runCampaign(ctx context.Context, campaignID string, def recipeDefinition) {
	campaign := m.GetCampaign(campaignID)
	if campaign == nil {
		return
	}

	m.updateCampaign(campaignID, func(current *Campaign) {
		current.Status = StatusRunning
	})
	m.recordEvent(campaignID, CampaignEvent{
		Timestamp: time.Now().UTC(),
		Kind:      EventStatus,
		Level:     "info",
		Message:   "Research loop started",
	})

	device := "cpu"
	if campaign.Platform == "amd" {
		device = "cuda"
	}

	baselineEval, err := m.runOfflineEval(ctx, campaignID, def, campaign.Baseline.ModelID, false, campaign.Overrides.DatasetOverride, filepath.Join(campaign.ArtifactDir, "baseline", "offline"), device, 0)
	if err != nil {
		m.failCampaign(campaignID, StatusFailed, fmt.Sprintf("baseline evaluation failed: %v", err))
		return
	}
	m.updateCampaign(campaignID, func(current *Campaign) {
		current.BaselineEval = baselineEval
	})
	m.recordEvent(campaignID, CampaignEvent{
		Timestamp: time.Now().UTC(),
		Kind:      EventMetric,
		Level:     "info",
		Message:   fmt.Sprintf("Baseline %s accuracy %.2f%%", def.Key, baselineEval.Accuracy*100),
		Percent:   10,
	})

	if runtimeDataset := runtimeDatasetForRecipe(def.Key, campaign.Overrides.DatasetOverride); runtimeDataset != "" && strings.TrimSpace(campaign.APIBase) != "" {
		runtimeEval, runtimeErr := m.runSignalEval(ctx, campaignID, runtimeDataset, campaign.APIBase, filepath.Join(campaign.ArtifactDir, "baseline", "runtime"), 0)
		if runtimeErr != nil {
			m.recordEvent(campaignID, CampaignEvent{
				Timestamp: time.Now().UTC(),
				Kind:      EventLog,
				Level:     "warn",
				Message:   fmt.Sprintf("Runtime baseline skipped: %v", runtimeErr),
				Percent:   15,
			})
		} else {
			m.updateCampaign(campaignID, func(current *Campaign) {
				current.RuntimeBaseline = runtimeEval
			})
		}
	}

	plans, err := buildTrialPlans(def, campaign.Budget.MaxTrials, campaign.Overrides.HyperparameterHints)
	if err != nil {
		m.failCampaign(campaignID, StatusFailed, err.Error())
		return
	}
	if len(plans) == 0 {
		m.failCampaign(campaignID, StatusFailed, "no trial plans generated")
		return
	}

	successfulTrials := 0
	bestAccuracy := -1.0
	for idx, plan := range plans {
		if err := ctx.Err(); err != nil {
			m.stopCampaign(campaignID, "Campaign stopped before all trials completed")
			return
		}

		trial := TrialResult{
			Index:         idx + 1,
			Name:          plan.Name,
			Status:        StatusRunning,
			StartedAt:     time.Now().UTC(),
			Params:        cloneMap(plan.Params),
			UseLoRA:       plan.UseLoRA,
			PrimaryMetric: def.PrimaryMetric,
		}
		m.updateCampaign(campaignID, func(current *Campaign) {
			current.Trials = append(current.Trials, trial)
		})
		m.recordEvent(campaignID, CampaignEvent{
			Timestamp:  time.Now().UTC(),
			Kind:       EventProgress,
			Level:      "info",
			Message:    fmt.Sprintf("Starting %s", plan.Name),
			Percent:    20 + int(float64(idx)/float64(len(plans))*60),
			TrialIndex: trial.Index,
		})

		trialDir := filepath.Join(campaign.ArtifactDir, "trials", plan.Name)
		if err := os.MkdirAll(trialDir, 0o755); err != nil {
			m.finishTrial(campaignID, idx, nil, "", nil, fmt.Errorf("create trial dir: %w", err))
			continue
		}

		modelPath, artifacts, trainErr := m.runTraining(ctx, campaignID, def, campaign, plan, trialDir, device, trial.Index)
		if trainErr != nil {
			m.finishTrial(campaignID, idx, nil, "", nil, trainErr)
			continue
		}

		evalResult, evalErr := m.runOfflineEval(ctx, campaignID, def, modelPath, plan.UseLoRA, campaign.Overrides.DatasetOverride, filepath.Join(trialDir, "eval"), device, trial.Index)
		if evalErr != nil {
			m.finishTrial(campaignID, idx, nil, modelPath, artifacts, evalErr)
			continue
		}

		evalResult.ImprovementPP = (evalResult.Accuracy - baselineEval.Accuracy) * 100
		successfulTrials++
		m.finishTrial(campaignID, idx, evalResult, modelPath, artifacts, nil)

		if evalResult.Accuracy > bestAccuracy {
			bestAccuracy = evalResult.Accuracy
			bestTrial := m.GetCampaign(campaignID).Trials[idx]
			m.updateCampaign(campaignID, func(current *Campaign) {
				current.BestTrial = &bestTrial
			})
			fragmentPath, fragmentErr := m.writeConfigFragment(campaignID, baselineEval, &bestTrial)
			if fragmentErr == nil {
				m.updateCampaign(campaignID, func(current *Campaign) {
					current.ConfigFragmentPath = fragmentPath
				})
			}
		}
	}

	if successfulTrials == 0 {
		m.failCampaign(campaignID, StatusFailed, "all trials failed")
		return
	}

	finalCampaign := m.GetCampaign(campaignID)
	if finalCampaign == nil || finalCampaign.BestTrial == nil || finalCampaign.BestTrial.Eval == nil {
		m.failCampaign(campaignID, StatusFailed, "no successful trial metrics available")
		return
	}

	improvement := finalCampaign.BestTrial.Eval.ImprovementPP
	if improvement < finalCampaign.SuccessThresholdPP {
		m.completeCampaign(campaignID, fmt.Sprintf("Loop finished; best trial improved by %.2fpp, below threshold %.2fpp", improvement, finalCampaign.SuccessThresholdPP))
		return
	}
	m.completeCampaign(campaignID, fmt.Sprintf("Loop finished; best trial improved by %.2fpp", improvement))
}

func (m *Manager) stopCampaign(id string, message string) {
	now := time.Now().UTC()
	m.updateCampaign(id, func(campaign *Campaign) {
		campaign.Status = StatusStopped
		campaign.CompletedAt = &now
	})
	m.recordEvent(id, CampaignEvent{
		Timestamp: now,
		Kind:      EventStatus,
		Level:     "warn",
		Message:   message,
		Percent:   100,
	})
	m.markTerminal(id)
}

func (m *Manager) markTerminal(id string) {
	m.mu.Lock()
	delete(m.cancelFns, id)
	m.mu.Unlock()
	select {
	case m.events <- StreamEvent{
		CampaignID: id,
		Event: CampaignEvent{
			Timestamp: time.Now().UTC(),
			Kind:      EventStatus,
			Level:     "info",
			Message:   "terminal",
			Percent:   100,
		},
		Terminal: true,
	}:
	default:
	}
}

func (m *Manager) finishTrial(
	campaignID string,
	index int,
	eval *MetricSnapshot,
	modelPath string,
	artifacts map[string]string,
	runErr error,
) {
	now := time.Now().UTC()
	m.updateCampaign(campaignID, func(campaign *Campaign) {
		if index >= len(campaign.Trials) {
			return
		}
		trial := campaign.Trials[index]
		if runErr != nil {
			trial.Status = StatusFailed
			trial.Error = runErr.Error()
		} else {
			trial.Status = StatusCompleted
			trial.Eval = eval
			trial.ModelPath = modelPath
			trial.Artifacts = artifacts
		}
		trial.CompletedAt = &now
		campaign.Trials[index] = trial
	})

	if runErr != nil {
		m.recordEvent(campaignID, CampaignEvent{
			Timestamp:  now,
			Kind:       EventLog,
			Level:      "error",
			Message:    runErr.Error(),
			TrialIndex: index + 1,
		})
		return
	}

	m.recordEvent(campaignID, CampaignEvent{
		Timestamp:  now,
		Kind:       EventMetric,
		Level:      "info",
		Message:    fmt.Sprintf("Trial %d accuracy %.2f%% (%+.2fpp)", index+1, eval.Accuracy*100, eval.ImprovementPP),
		TrialIndex: index + 1,
	})
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

func (m *Manager) runDomainTraining(ctx context.Context, campaignID string, _ *Campaign, plan trialPlan, trialDir, device string, trialIndex int) (string, map[string]string, error) {
	scriptDir := filepath.Join(m.repoRoot, "src", "training", "model_classifier", "classifier_model_fine_tuning_lora")
	outputDir := filepath.Join(trialDir, "domain-lora")
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
	case "fact-check", "jailbreak", "domain":
		common["model"] = struct{}{}
	}
	if recipeKey == "domain" {
		common["max_samples"] = struct{}{}
		common["enable_feature_alignment"] = struct{}{}
		common["alignment_weight"] = struct{}{}
	}
	return common
}

func runtimeDatasetForRecipe(recipeKey string, override string) string {
	if strings.TrimSpace(override) != "" && !strings.Contains(override, "/") && !strings.Contains(override, ".") {
		return override
	}
	switch recipeKey {
	case "feedback":
		return "feedback-en"
	case "fact-check":
		return "fact-check-en"
	case "domain":
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
	if recipeKey == "domain" && !strings.Contains(override, ".json") && !strings.Contains(override, ".csv") {
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
