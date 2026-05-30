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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

func cloneUserPreferenceStates(states map[string]*UserPreferenceState) map[string]*UserPreferenceState {
	cloned := make(map[string]*UserPreferenceState, len(states))
	for userID, state := range states {
		if state == nil {
			continue
		}

		next := *state
		next.Interactions = append([]InteractionRecord(nil), state.Interactions...)
		next.ModelPreferences = make(map[string]float64, len(state.ModelPreferences))
		for model, score := range state.ModelPreferences {
			next.ModelPreferences[model] = score
		}
		cloned[userID] = &next
	}
	return cloned
}

func saveGMTRouterState(path string, states map[string]*UserPreferenceState) error {
	if path == "" {
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create GMTRouter state directory: %w", err)
	}

	data, err := json.MarshalIndent(states, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal GMTRouter state: %w", err)
	}

	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("failed to write GMTRouter temp state: %w", err)
	}

	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to replace GMTRouter state: %w", err)
	}
	return nil
}

func loadGMTRouterState(path string) (map[string]*UserPreferenceState, bool, error) {
	if path == "" {
		return nil, false, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("failed to read GMTRouter state: %w", err)
	}

	var states map[string]*UserPreferenceState
	if err := json.Unmarshal(data, &states); err != nil {
		backupPath := path + ".corrupted"
		if backupErr := os.WriteFile(backupPath, data, 0o644); backupErr != nil {
			return nil, false, fmt.Errorf("failed to parse GMTRouter state and write backup: %w", err)
		}
		return nil, false, fmt.Errorf("failed to parse GMTRouter state (backup saved to %s): %w", backupPath, err)
	}

	normalizeUserPreferenceStates(states)
	return states, true, nil
}

func normalizeUserPreferenceStates(states map[string]*UserPreferenceState) {
	for userID, state := range states {
		if state == nil {
			delete(states, userID)
			continue
		}
		if state.UserID == "" {
			state.UserID = userID
		}
		if state.Interactions == nil {
			state.Interactions = make([]InteractionRecord, 0)
		}
		if state.ModelPreferences == nil {
			state.ModelPreferences = make(map[string]float64)
		}
		if state.TotalInteractions < len(state.Interactions) {
			state.TotalInteractions = len(state.Interactions)
		}
	}
}
