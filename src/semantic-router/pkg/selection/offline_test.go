/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func boolPtr(v bool) *bool { return &v }

func offlineWinRecord(turnIndex int, selectedModel, opponentModel string, timestamp time.Time) OfflineDatasetRecord {
	return OfflineDatasetRecord{
		SessionID:       "s1",
		TurnIndex:       turnIndex,
		SelectedModel:   selectedModel,
		CandidateModels: []string{"gpt-4o", "gpt-4o-mini"},
		SelectionMethod: "rl_driven",
		Outcome: OfflineOutcome{
			Won:           boolPtr(true),
			OpponentModel: opponentModel,
		},
		Timestamp: timestamp,
	}
}

func gpt4oPolicyVersion(id, parentID, source string, alpha, beta float64) *PolicyVersion {
	return &PolicyVersion{
		ID:        id,
		ParentID:  parentID,
		CreatedAt: time.Now(),
		Source:    source,
		Status:    PolicyStatusCandidate,
		Weights: PolicyWeights{
			Global: map[string]*ModelPreference{
				"gpt-4o": {Model: "gpt-4o", Distribution: BetaDistribution{Alpha: alpha, Beta: beta}},
			},
		},
	}
}

func emptyPolicyVersion(id, source string) *PolicyVersion {
	return &PolicyVersion{
		ID:      id,
		Source:  source,
		Status:  PolicyStatusCandidate,
		Weights: PolicyWeights{Global: map[string]*ModelPreference{}},
	}
}

func savePolicyVersion(t *testing.T, store *PolicyVersionStore, version *PolicyVersion) {
	t.Helper()
	if err := store.Save(version); err != nil {
		t.Fatalf("save %s: %v", version.ID, err)
	}
}

func activatePolicyVersion(t *testing.T, store *PolicyVersionStore, versionID string) {
	t.Helper()
	if err := store.Activate(versionID); err != nil {
		t.Fatalf("activate %s: %v", versionID, err)
	}
}

func shadowPolicyVersion(t *testing.T, store *PolicyVersionStore, versionID string) {
	t.Helper()
	if err := store.Shadow(versionID); err != nil {
		t.Fatalf("shadow %s: %v", versionID, err)
	}
}

func requireManifestVersions(t *testing.T, store *PolicyVersionStore, activeVersion, shadowVersion string) {
	t.Helper()
	manifest, err := store.LoadManifest()
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if manifest.ActiveVersion != activeVersion {
		t.Errorf("expected active version %s, got %s", activeVersion, manifest.ActiveVersion)
	}
	if manifest.ShadowVersion != shadowVersion {
		t.Errorf("expected shadow version %s, got %s", shadowVersion, manifest.ShadowVersion)
	}
}

func requirePolicyStatus(t *testing.T, store *PolicyVersionStore, versionID string, status PolicyStatus) {
	t.Helper()
	version, err := store.Load(versionID)
	if err != nil {
		t.Fatalf("load %s: %v", versionID, err)
	}
	if version.Status != status {
		t.Errorf("expected %s status %s, got %s", versionID, status, version.Status)
	}
}

func TestOfflineUpdater_EmptyDataset(t *testing.T) {
	updater := NewOfflineUpdater(nil)
	_, err := updater.Update(context.Background(), nil, nil)
	if err == nil {
		t.Fatal("expected error for nil dataset")
	}
	_, err = updater.Update(context.Background(), &OfflineDataset{}, nil)
	if err == nil {
		t.Fatal("expected error for empty dataset")
	}
}

func TestOfflineUpdater_BasicWinLoss(t *testing.T) {
	updater := NewOfflineUpdater(DefaultOfflineUpdaterConfig())
	now := time.Now()

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-24 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			offlineWinRecord(0, "gpt-4o", "gpt-4o-mini", now.Add(-23*time.Hour)),
			offlineWinRecord(1, "gpt-4o", "gpt-4o-mini", now.Add(-22*time.Hour)),
			offlineWinRecord(2, "gpt-4o-mini", "gpt-4o", now.Add(-21*time.Hour)),
			offlineWinRecord(3, "gpt-4o-mini", "gpt-4o", now.Add(-20*time.Hour)),
			offlineWinRecord(4, "gpt-4o-mini", "gpt-4o", now.Add(-19*time.Hour)),
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if version.Status != PolicyStatusCandidate {
		t.Errorf("expected candidate status, got %s", version.Status)
	}
	if version.Source != "offline_batch" {
		t.Errorf("expected offline_batch source, got %s", version.Source)
	}

	// gpt-4o: 2 wins (as selected), 3 losses (as opponent) => alpha=1+2=3, beta=1+3=4
	gpt4o := version.Weights.Global["gpt-4o"]
	if gpt4o == nil {
		t.Fatal("expected gpt-4o in global weights")
	}
	if gpt4o.Distribution.Alpha != 3.0 {
		t.Errorf("expected gpt-4o alpha=3.0, got %.1f", gpt4o.Distribution.Alpha)
	}
	if gpt4o.Distribution.Beta != 4.0 {
		t.Errorf("expected gpt-4o beta=4.0, got %.1f", gpt4o.Distribution.Beta)
	}

	// gpt-4o-mini: 3 wins (as selected), 2 losses (as opponent) => alpha=1+3=4, beta=1+2=3
	mini := version.Weights.Global["gpt-4o-mini"]
	if mini == nil {
		t.Fatal("expected gpt-4o-mini in global weights")
	}
	if mini.Distribution.Alpha != 4.0 {
		t.Errorf("expected gpt-4o-mini alpha=4.0, got %.1f", mini.Distribution.Alpha)
	}
	if mini.Distribution.Beta != 3.0 {
		t.Errorf("expected gpt-4o-mini beta=3.0, got %.1f", mini.Distribution.Beta)
	}
}

func TestOfflineUpdater_TransitionPenalty(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.TransitionPenaltyWeight = 0.5
	cfg.MinRecordsPerModel = 1
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	dataset := &OfflineDataset{
		Version:         1,
		CreatedAt:       now,
		WindowStart:     now.Add(-1 * time.Hour),
		WindowEnd:       now,
		SessionCount:    1,
		TransitionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				TurnIndex:       1,
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o", "claude-3"},
				SelectionMethod: "rl_driven",
				Outcome:         OfflineOutcome{Won: boolPtr(true), OpponentModel: "claude-3"},
				TransitionEvidence: &OfflineTransitionEvidence{
					PreviousModel: "claude-3",
					CacheWarmth:   0.8,
					CacheWarmthOK: true,
				},
				Timestamp: now.Add(-30 * time.Minute),
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// claude-3 gets: 1 loss (as opponent) + 0.5 transition penalty
	claude := version.Weights.Global["claude-3"]
	if claude == nil {
		t.Fatal("expected claude-3 in global weights")
	}
	// beta should be 1 (prior) + 1 (opponent loss) + 0.5 (transition) = 2.5
	if claude.Distribution.Beta != 2.5 {
		t.Errorf("expected claude-3 beta=2.5, got %.1f", claude.Distribution.Beta)
	}
}

func TestOfflineUpdater_ParentWeightBlending(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.MinRecordsPerModel = 10 // High threshold so sparse model keeps parent weights
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	parentWeights := &PolicyWeights{
		Global: map[string]*ModelPreference{
			"gpt-4o": {
				Model:             "gpt-4o",
				Distribution:      BetaDistribution{Alpha: 10.0, Beta: 5.0},
				TotalInteractions: 15,
			},
			"claude-3": {
				Model:             "claude-3",
				Distribution:      BetaDistribution{Alpha: 8.0, Beta: 8.0},
				TotalInteractions: 16,
			},
		},
	}

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-1 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o"},
				Outcome:         OfflineOutcome{Won: boolPtr(true)},
				Timestamp:       now,
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, parentWeights)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// gpt-4o has only 1 record < MinRecordsPerModel=10, should keep parent weights
	gpt4o := version.Weights.Global["gpt-4o"]
	if gpt4o.Distribution.Alpha != 10.0 || gpt4o.Distribution.Beta != 5.0 {
		t.Errorf("expected parent weights for sparse model, got alpha=%.1f beta=%.1f",
			gpt4o.Distribution.Alpha, gpt4o.Distribution.Beta)
	}

	// claude-3 had no new records, should be carried forward from parent
	claude := version.Weights.Global["claude-3"]
	if claude == nil {
		t.Fatal("expected claude-3 carried forward from parent")
	}
	if claude.Distribution.Alpha != 8.0 {
		t.Errorf("expected claude-3 alpha=8.0, got %.1f", claude.Distribution.Alpha)
	}
}

func TestOfflineUpdater_CategoryWeights(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.MinRecordsPerModel = 1
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-1 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				DecisionName:    "math",
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o"},
				Outcome:         OfflineOutcome{Won: boolPtr(true)},
				Timestamp:       now,
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if version.Weights.Category == nil || version.Weights.Category["math"] == nil {
		t.Fatal("expected math category weights")
	}
	mathGpt4o := version.Weights.Category["math"]["gpt-4o"]
	if mathGpt4o == nil {
		t.Fatal("expected gpt-4o in math category")
	}
	if mathGpt4o.Distribution.Alpha != 2.0 {
		t.Errorf("expected alpha=2.0, got %.1f", mathGpt4o.Distribution.Alpha)
	}
}

func TestOfflineUpdater_ImplicitFeedbackWeight(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.ImplicitFeedbackWeight = 0.5
	cfg.MinRecordsPerModel = 1
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-1 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o"},
				Outcome: OfflineOutcome{
					Won:                boolPtr(true),
					FeedbackType:       "implicit_satisfied",
					FeedbackConfidence: 0.8,
				},
				Timestamp: now,
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gpt4o := version.Weights.Global["gpt-4o"]
	// weight = 0.5 * 0.8 = 0.4, so alpha = 1 + 0.4 = 1.4
	if gpt4o.Distribution.Alpha != 1.4 {
		t.Errorf("expected alpha=1.4, got %.1f", gpt4o.Distribution.Alpha)
	}
}

func TestOfflineUpdater_ExplicitFeedbackFullWeight(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.ImplicitFeedbackWeight = 0.5
	cfg.MinRecordsPerModel = 1
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-1 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o"},
				Outcome: OfflineOutcome{
					Won:          boolPtr(true),
					FeedbackType: "explicit",
				},
				Timestamp: now,
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gpt4o := version.Weights.Global["gpt-4o"]
	// Explicit feedback should get full weight (1.0), so alpha = 1 + 1.0 = 2.0
	if gpt4o.Distribution.Alpha != 2.0 {
		t.Errorf("expected alpha=2.0 for explicit feedback, got %.1f", gpt4o.Distribution.Alpha)
	}
}

func TestCompareVersions(t *testing.T) {
	a := &PolicyVersion{
		ID: "v1",
		Weights: PolicyWeights{
			Global: map[string]*ModelPreference{
				"gpt-4o": {Model: "gpt-4o", Distribution: BetaDistribution{Alpha: 10, Beta: 5}},
				"claude": {Model: "claude", Distribution: BetaDistribution{Alpha: 5, Beta: 5}},
			},
		},
	}
	b := &PolicyVersion{
		ID: "v2",
		Weights: PolicyWeights{
			Global: map[string]*ModelPreference{
				"gpt-4o": {Model: "gpt-4o", Distribution: BetaDistribution{Alpha: 12, Beta: 4}},
				"llama":  {Model: "llama", Distribution: BetaDistribution{Alpha: 3, Beta: 7}},
			},
		},
	}

	diffs := CompareVersions(a, b)
	if len(diffs) != 3 {
		t.Fatalf("expected 3 model diffs, got %d", len(diffs))
	}

	// Check sorted order: claude, gpt-4o, llama
	if diffs[0].Model != "claude" {
		t.Errorf("expected first diff to be claude, got %s", diffs[0].Model)
	}
	if diffs[1].Model != "gpt-4o" {
		t.Errorf("expected second diff to be gpt-4o, got %s", diffs[1].Model)
	}

	// gpt-4o: mean_a = 10/15 ≈ 0.667, mean_b = 12/16 = 0.75
	if diffs[1].MeanDelta < 0.08 || diffs[1].MeanDelta > 0.09 {
		t.Errorf("expected gpt-4o mean_delta ≈ 0.083, got %.3f", diffs[1].MeanDelta)
	}
}

func TestPolicyVersionStore_SaveLoadActivate(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "policy-version-test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

	store, err := NewPolicyVersionStore(filepath.Join(tmpDir, "policies"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	v1 := gpt4oPolicyVersion("v1", "", "manual", 5, 3)

	savePolicyVersion(t, store, v1)

	loaded, err := store.Load("v1")
	if err != nil {
		t.Fatalf("load v1: %v", err)
	}
	if loaded.ID != "v1" {
		t.Errorf("expected v1, got %s", loaded.ID)
	}
	if loaded.Weights.Global["gpt-4o"].Distribution.Alpha != 5 {
		t.Errorf("expected alpha=5, got %.1f", loaded.Weights.Global["gpt-4o"].Distribution.Alpha)
	}

	// Activate v1
	activatePolicyVersion(t, store, "v1")
	requireManifestVersions(t, store, "v1", "")

	// Save v2 and shadow it
	v2 := gpt4oPolicyVersion("v2", "v1", "offline_batch", 8, 4)
	savePolicyVersion(t, store, v2)
	shadowPolicyVersion(t, store, "v2")
	requireManifestVersions(t, store, "v1", "v2")

	// Promote v2 to active (v1 gets retired)
	activatePolicyVersion(t, store, "v2")
	requireManifestVersions(t, store, "v2", "")

	// v1 should be retired
	requirePolicyStatus(t, store, "v1", PolicyStatusRetired)
}

func TestPolicyVersionStore_Revert(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "policy-revert-test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

	store, err := NewPolicyVersionStore(filepath.Join(tmpDir, "policies"))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	v1 := emptyPolicyVersion("v1", "manual")
	v2 := emptyPolicyVersion("v2", "offline_batch")

	_ = store.Save(v1)
	_ = store.Save(v2)
	_ = store.Activate("v1")
	_ = store.Activate("v2")

	// Revert to v1
	if err := store.Revert("v1"); err != nil {
		t.Fatalf("revert: %v", err)
	}

	manifest, _ := store.LoadManifest()
	if manifest.ActiveVersion != "v1" {
		t.Errorf("expected v1 after revert, got %s", manifest.ActiveVersion)
	}

	v2Loaded, _ := store.Load("v2")
	if v2Loaded.Status != PolicyStatusRetired {
		t.Errorf("expected v2 retired after revert, got %s", v2Loaded.Status)
	}
}

func TestOfflineUpdater_ServerError(t *testing.T) {
	cfg := DefaultOfflineUpdaterConfig()
	cfg.MinRecordsPerModel = 1
	updater := NewOfflineUpdater(cfg)
	now := time.Now()

	dataset := &OfflineDataset{
		Version:      1,
		CreatedAt:    now,
		WindowStart:  now.Add(-1 * time.Hour),
		WindowEnd:    now,
		SessionCount: 1,
		Records: []OfflineDatasetRecord{
			{
				SessionID:       "s1",
				SelectedModel:   "gpt-4o",
				CandidateModels: []string{"gpt-4o"},
				Outcome:         OfflineOutcome{ResponseStatus: 500},
				Timestamp:       now,
			},
		},
	}

	version, err := updater.Update(context.Background(), dataset, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gpt4o := version.Weights.Global["gpt-4o"]
	// Server error adds 0.5 to beta: beta = 1 (prior) + 0.5 = 1.5
	if gpt4o.Distribution.Beta != 1.5 {
		t.Errorf("expected beta=1.5 for server error, got %.1f", gpt4o.Distribution.Beta)
	}
}
