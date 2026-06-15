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
	"path/filepath"
	"testing"
	"time"
)

func TestRLDrivenSelector_StoragePersistsUserAndSessionState(t *testing.T) {
	storagePath := filepath.Join(t.TempDir(), "rl_state.json")
	cfg := &RLDrivenConfig{
		UseThompsonSampling:   true,
		EnablePersonalization: true,
		PersonalizationBlend:  0.7,
		SessionContextWeight:  0.3,
		StoragePath:           storagePath,
		AutoSaveInterval:      "1h",
	}

	selector := NewRLDrivenSelector(cfg)
	err := selector.UpdateFeedback(context.Background(), &Feedback{
		WinnerModel:  "model-a",
		LoserModel:   "model-b",
		UserID:       "user/123",
		SessionID:    "session:abc",
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}
	if err := selector.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reloaded := NewRLDrivenSelector(cfg)
	defer func() {
		if err := reloaded.Close(); err != nil {
			t.Fatalf("reloaded Close() error = %v", err)
		}
	}()

	requireRLPreferenceBias(t, reloaded, "user/123", "model-a", true)
	requireRLPreferenceBias(t, reloaded, "user/123", "model-b", false)
	requireRLSessionStats(t, reloaded, "session:abc", "model-a", 1, 1)
	requireRLSessionStats(t, reloaded, "session:abc", "model-b", 1, 0)
}

func requireRLPreferenceBias(
	t *testing.T,
	selector *RLDrivenSelector,
	userID string,
	model string,
	wantAlphaHigher bool,
) {
	t.Helper()

	pref := selector.getUserPreference(userID, model)
	if pref == nil {
		t.Fatalf("reloaded user preference for %s/%s is nil", userID, model)
	}
	alpha := pref.Distribution.Alpha
	beta := pref.Distribution.Beta
	if wantAlphaHigher && alpha <= beta {
		t.Fatalf("reloaded user %s preference alpha=%v beta=%v, want alpha > beta", model, alpha, beta)
	}
	if !wantAlphaHigher && beta <= alpha {
		t.Fatalf("reloaded user %s preference alpha=%v beta=%v, want beta > alpha", model, alpha, beta)
	}
}

func requireRLSessionStats(
	t *testing.T,
	selector *RLDrivenSelector,
	sessionID string,
	model string,
	wantSelections int,
	wantSuccesses int,
) {
	t.Helper()
	selector.sessionMu.RLock()
	defer selector.sessionMu.RUnlock()

	statsByModel := selector.sessionContext[sessionID]
	if statsByModel == nil {
		t.Fatalf("reloaded session context for %s is nil", sessionID)
	}
	stats := statsByModel[model]
	if stats == nil || stats.Selections != wantSelections || stats.Successes != wantSuccesses {
		t.Fatalf("%s session stats = %#v, want selections=%d successes=%d",
			model, stats, wantSelections, wantSuccesses)
	}
}
