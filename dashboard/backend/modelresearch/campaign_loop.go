package modelresearch

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (m *Manager) runCampaign(ctx context.Context, campaignID string, def recipeDefinition) {
	campaign := m.startCampaignLoop(campaignID)
	if campaign == nil {
		return
	}

	device, err := m.resolveCampaignDevice(ctx, campaignID, campaign)
	if err != nil {
		m.failCampaign(campaignID, StatusFailed, err.Error())
		return
	}
	baselineEval, err := m.runCampaignBaseline(ctx, campaignID, def, campaign, device)
	if err != nil {
		m.failCampaign(campaignID, StatusFailed, fmt.Sprintf("baseline evaluation failed: %v", err))
		return
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

	successfulTrials, stopped := m.runTrialPlans(ctx, campaignID, def, campaign, plans, baselineEval, device)
	if stopped {
		return
	}
	if successfulTrials == 0 {
		m.failCampaign(campaignID, StatusFailed, "all trials failed")
		return
	}

	m.completeCampaignFromBestTrial(campaignID)
}

func (m *Manager) startCampaignLoop(campaignID string) *Campaign {
	campaign := m.GetCampaign(campaignID)
	if campaign == nil {
		return nil
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
	return campaign
}

func campaignDevice(campaign *Campaign) string {
	if campaign != nil && campaign.Platform == "amd" {
		return "cuda"
	}
	return "cpu"
}

func (m *Manager) resolveCampaignDevice(ctx context.Context, campaignID string, campaign *Campaign) (string, error) {
	requestedDevice := campaignDevice(campaign)
	if requestedDevice != "cuda" {
		return "cpu", nil
	}

	available, err := m.pythonCUDAAvailable(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to probe Python runtime for GPU availability: %w", err)
	}
	if available {
		return "cuda", nil
	}

	m.recordEvent(campaignID, CampaignEvent{
		Timestamp: time.Now().UTC(),
		Kind:      EventLog,
		Level:     "warn",
		Message:   "GPU runtime was not detected in the current worker; the campaign will run on CPU instead.",
		Percent:   5,
	})
	return "cpu", nil
}

func (m *Manager) pythonCUDAAvailable(ctx context.Context) (bool, error) {
	var lines []string
	spec := commandSpec{
		Dir:  firstNonEmpty(m.repoRoot, "."),
		Name: m.pythonPath,
		Args: []string{"-c", "import torch; print('1' if torch.cuda.is_available() else '0')"},
	}
	err := m.runCommand(ctx, spec, func(stream, line string) {
		if strings.EqualFold(stream, "stdout") {
			lines = append(lines, strings.TrimSpace(line))
		}
	})
	if err != nil {
		return false, err
	}
	for _, line := range lines {
		if line == "1" {
			return true, nil
		}
		if line == "0" {
			return false, nil
		}
	}
	return false, nil
}

func (m *Manager) runCampaignBaseline(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	campaign *Campaign,
	device string,
) (*MetricSnapshot, error) {
	baselineEval, err := m.runOfflineEval(
		ctx,
		campaignID,
		def,
		campaign.Baseline.ModelID,
		false,
		campaign.Overrides.DatasetOverride,
		filepath.Join(campaign.ArtifactDir, "baseline", "offline"),
		device,
		0,
	)
	if err != nil {
		return nil, err
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
	m.maybeRunRuntimeBaseline(ctx, campaignID, def, campaign)
	return baselineEval, nil
}

func (m *Manager) maybeRunRuntimeBaseline(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	campaign *Campaign,
) {
	runtimeDataset := runtimeDatasetForRecipe(def.Key, campaign.Overrides.DatasetOverride)
	if runtimeDataset == "" || strings.TrimSpace(campaign.APIBase) == "" {
		return
	}

	runtimeEval, err := m.runSignalEval(
		ctx,
		campaignID,
		runtimeDataset,
		campaign.APIBase,
		filepath.Join(campaign.ArtifactDir, "baseline", "runtime"),
		0,
	)
	if err != nil {
		m.recordEvent(campaignID, CampaignEvent{
			Timestamp: time.Now().UTC(),
			Kind:      EventLog,
			Level:     "warn",
			Message:   fmt.Sprintf("Runtime baseline skipped: %v", err),
			Percent:   15,
		})
		return
	}

	m.updateCampaign(campaignID, func(current *Campaign) {
		current.RuntimeBaseline = runtimeEval
	})
}

func (m *Manager) runTrialPlans(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	campaign *Campaign,
	plans []trialPlan,
	baselineEval *MetricSnapshot,
	device string,
) (int, bool) {
	successfulTrials := 0
	bestAccuracy := -1.0

	for idx, plan := range plans {
		if err := ctx.Err(); err != nil {
			m.stopCampaign(campaignID, "Campaign stopped before all trials completed")
			return successfulTrials, true
		}

		trial := m.enqueueTrial(campaignID, def, plan, idx, len(plans))
		evalResult, modelPath, artifacts, err := m.runTrialPlan(
			ctx,
			campaignID,
			def,
			campaign,
			plan,
			trial,
			baselineEval,
			device,
		)
		if err != nil {
			m.finishTrial(campaignID, idx, nil, modelPath, artifacts, err)
			continue
		}

		successfulTrials++
		m.finishTrial(campaignID, idx, evalResult, modelPath, artifacts, nil)
		bestAccuracy = m.promoteBestTrial(campaignID, idx, baselineEval, evalResult.Accuracy, bestAccuracy)
	}

	return successfulTrials, false
}

func (m *Manager) enqueueTrial(
	campaignID string,
	def recipeDefinition,
	plan trialPlan,
	index int,
	total int,
) TrialResult {
	trial := TrialResult{
		Index:         index + 1,
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
		Percent:    20 + int(float64(index)/float64(total)*60),
		TrialIndex: trial.Index,
	})
	return trial
}

func (m *Manager) runTrialPlan(
	ctx context.Context,
	campaignID string,
	def recipeDefinition,
	campaign *Campaign,
	plan trialPlan,
	trial TrialResult,
	baselineEval *MetricSnapshot,
	device string,
) (*MetricSnapshot, string, map[string]string, error) {
	trialDir := filepath.Join(campaign.ArtifactDir, "trials", plan.Name)
	if err := os.MkdirAll(trialDir, 0o755); err != nil {
		return nil, "", nil, fmt.Errorf("create trial dir: %w", err)
	}

	modelPath, artifacts, err := m.runTraining(
		ctx,
		campaignID,
		def,
		campaign,
		plan,
		trialDir,
		device,
		trial.Index,
	)
	if err != nil {
		return nil, "", nil, err
	}

	evalResult, err := m.runOfflineEval(
		ctx,
		campaignID,
		def,
		modelPath,
		plan.UseLoRA,
		campaign.Overrides.DatasetOverride,
		filepath.Join(trialDir, "eval"),
		device,
		trial.Index,
	)
	if err != nil {
		return nil, modelPath, artifacts, err
	}

	evalResult.ImprovementPP = (evalResult.Accuracy - baselineEval.Accuracy) * 100
	return evalResult, modelPath, artifacts, nil
}

func (m *Manager) promoteBestTrial(
	campaignID string,
	index int,
	baselineEval *MetricSnapshot,
	accuracy float64,
	bestAccuracy float64,
) float64 {
	if accuracy <= bestAccuracy {
		return bestAccuracy
	}

	current := m.GetCampaign(campaignID)
	if current == nil || index >= len(current.Trials) {
		return bestAccuracy
	}

	bestTrial := current.Trials[index]
	m.updateCampaign(campaignID, func(campaign *Campaign) {
		campaign.BestTrial = &bestTrial
	})
	if fragmentPath, err := m.writeConfigFragment(campaignID, baselineEval, &bestTrial); err == nil {
		m.updateCampaign(campaignID, func(campaign *Campaign) {
			campaign.ConfigFragmentPath = fragmentPath
		})
	}
	return accuracy
}

func (m *Manager) completeCampaignFromBestTrial(campaignID string) {
	finalCampaign := m.GetCampaign(campaignID)
	if finalCampaign == nil || finalCampaign.BestTrial == nil || finalCampaign.BestTrial.Eval == nil {
		m.failCampaign(campaignID, StatusFailed, "no successful trial metrics available")
		return
	}

	improvement := finalCampaign.BestTrial.Eval.ImprovementPP
	if improvement < finalCampaign.SuccessThresholdPP {
		m.completeCampaign(
			campaignID,
			fmt.Sprintf(
				"Loop finished; best trial improved by %.2fpp, below threshold %.2fpp",
				improvement,
				finalCampaign.SuccessThresholdPP,
			),
		)
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
