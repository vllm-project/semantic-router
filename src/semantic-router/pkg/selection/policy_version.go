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
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PolicyVersion identifies a snapshot of learned weights that can be activated,
// shadowed, compared, or reverted independently from the runtime code path.
type PolicyVersion struct {
	// ID is a unique identifier (e.g. "v3", "offline-2026-05-10-1400").
	ID string `json:"id"`

	// ParentID is the version this was derived from (empty for the initial version).
	ParentID string `json:"parent_id,omitempty"`

	// CreatedAt is when this version was produced.
	CreatedAt time.Time `json:"created_at"`

	// Source describes how the weights were produced.
	// Values: "online" (runtime UpdateFeedback), "offline_batch", "manual", "import".
	Source string `json:"source"`

	// DatasetWindow records the time window of data used for offline training.
	DatasetWindow *TimeWindow `json:"dataset_window,omitempty"`

	// TrainingStats summarizes what went into producing this version.
	TrainingStats *TrainingStats `json:"training_stats,omitempty"`

	// Status is the rollout state of this version.
	// Values: "candidate", "shadow", "active", "retired".
	Status PolicyStatus `json:"status"`

	// Weights holds the learned Beta distribution parameters per scope.
	Weights PolicyWeights `json:"weights"`
}

// TimeWindow represents a time range.
type TimeWindow struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// TrainingStats summarizes what data was used to produce a policy version.
type TrainingStats struct {
	RecordCount     int `json:"record_count"`
	SessionCount    int `json:"session_count"`
	TransitionCount int `json:"transition_count"`
	FeedbackCount   int `json:"feedback_count"`
}

// PolicyStatus is the rollout state of a policy version.
type PolicyStatus string

const (
	// PolicyStatusCandidate means the version was produced but not yet evaluated.
	PolicyStatusCandidate PolicyStatus = "candidate"

	// PolicyStatusShadow means the version is being evaluated in shadow mode
	// alongside the active version, without affecting real routing decisions.
	PolicyStatusShadow PolicyStatus = "shadow"

	// PolicyStatusActive means this version drives real routing decisions.
	PolicyStatusActive PolicyStatus = "active"

	// PolicyStatusRetired means this version was deactivated (kept for audit).
	PolicyStatusRetired PolicyStatus = "retired"
)

// PolicyWeights holds learned parameters at each scope level.
type PolicyWeights struct {
	// Global preferences (model -> preference).
	Global map[string]*ModelPreference `json:"global,omitempty"`

	// Category preferences (category -> model -> preference).
	Category map[string]map[string]*ModelPreference `json:"category,omitempty"`

	// User preferences (user_id -> model -> preference).
	User map[string]map[string]*ModelPreference `json:"user,omitempty"`
}

// PolicyVersionStore manages versioned policy weights on disk.
// Versions are stored as individual JSON files in a directory, with a
// manifest tracking the active and shadow versions.
type PolicyVersionStore struct {
	dir string
	mu  sync.RWMutex
}

// PolicyManifest tracks which versions are active and shadowed.
type PolicyManifest struct {
	ActiveVersion string          `json:"active_version"`
	ShadowVersion string          `json:"shadow_version,omitempty"`
	Versions      []PolicyVersion `json:"versions"`
}

// NewPolicyVersionStore creates a store rooted at dir.
func NewPolicyVersionStore(dir string) (*PolicyVersionStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create policy version directory: %w", err)
	}
	return &PolicyVersionStore{dir: dir}, nil
}

// Save persists a policy version to disk.
func (s *PolicyVersionStore) Save(version *PolicyVersion) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := json.MarshalIndent(version, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal policy version: %w", err)
	}

	path := s.versionPath(version.ID)
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("write policy version: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename policy version: %w", err)
	}

	logging.Infof("[PolicyVersionStore] Saved version %s (status=%s)", version.ID, version.Status)
	return nil
}

// Load reads a policy version from disk.
func (s *PolicyVersionStore) Load(versionID string) (*PolicyVersion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := os.ReadFile(s.versionPath(versionID))
	if err != nil {
		return nil, fmt.Errorf("read policy version %s: %w", versionID, err)
	}

	var version PolicyVersion
	if err := json.Unmarshal(data, &version); err != nil {
		return nil, fmt.Errorf("parse policy version %s: %w", versionID, err)
	}
	return &version, nil
}

// LoadManifest reads the manifest that tracks active/shadow versions.
func (s *PolicyVersionStore) LoadManifest() (*PolicyManifest, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := os.ReadFile(s.manifestPath())
	if err != nil {
		if os.IsNotExist(err) {
			return &PolicyManifest{}, nil
		}
		return nil, fmt.Errorf("read manifest: %w", err)
	}

	var manifest PolicyManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	return &manifest, nil
}

// SaveManifest writes the manifest atomically.
func (s *PolicyVersionStore) SaveManifest(manifest *PolicyManifest) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}

	path := s.manifestPath()
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename manifest: %w", err)
	}
	return nil
}

// Activate promotes a version to active, retiring the previous active version.
func (s *PolicyVersionStore) Activate(versionID string) error {
	manifest, err := s.LoadManifest()
	if err != nil {
		return err
	}

	// Retire the current active version
	if manifest.ActiveVersion != "" && manifest.ActiveVersion != versionID {
		prev, loadErr := s.Load(manifest.ActiveVersion)
		if loadErr == nil {
			prev.Status = PolicyStatusRetired
			if saveErr := s.Save(prev); saveErr != nil {
				logging.Warnf("[PolicyVersionStore] Failed to retire previous version %s: %v", prev.ID, saveErr)
			}
		}
	}

	version, err := s.Load(versionID)
	if err != nil {
		return err
	}
	version.Status = PolicyStatusActive
	if err := s.Save(version); err != nil {
		return err
	}

	manifest.ActiveVersion = versionID
	if manifest.ShadowVersion == versionID {
		manifest.ShadowVersion = ""
	}
	return s.SaveManifest(manifest)
}

// Shadow sets a version as the shadow for comparison against the active version.
func (s *PolicyVersionStore) Shadow(versionID string) error {
	manifest, err := s.LoadManifest()
	if err != nil {
		return err
	}

	version, err := s.Load(versionID)
	if err != nil {
		return err
	}
	version.Status = PolicyStatusShadow
	if err := s.Save(version); err != nil {
		return err
	}

	manifest.ShadowVersion = versionID
	return s.SaveManifest(manifest)
}

// Revert retires the current active version and promotes the specified version.
func (s *PolicyVersionStore) Revert(versionID string) error {
	return s.Activate(versionID)
}

func (s *PolicyVersionStore) versionPath(id string) string {
	return filepath.Join(s.dir, fmt.Sprintf("policy_%s.json", id))
}

func (s *PolicyVersionStore) manifestPath() string {
	return filepath.Join(s.dir, "manifest.json")
}

// ShadowComparison records the difference between active and shadow policy
// decisions for a single routing event, enabling operators to evaluate
// offline-trained weights before promotion.
type ShadowComparison struct {
	// SessionID is the session context of the comparison.
	SessionID string `json:"session_id"`

	// ActiveVersion is the version that made the live decision.
	ActiveVersion string `json:"active_version"`

	// ShadowVersion is the version being evaluated.
	ShadowVersion string `json:"shadow_version"`

	// ActiveModel is the model selected by the active policy.
	ActiveModel string `json:"active_model"`

	// ShadowModel is the model the shadow policy would have selected.
	ShadowModel string `json:"shadow_model"`

	// Agreed is true when both policies chose the same model.
	Agreed bool `json:"agreed"`

	// ActiveScore is the active policy's score for its chosen model.
	ActiveScore float64 `json:"active_score"`

	// ShadowScore is the shadow policy's score for its chosen model.
	ShadowScore float64 `json:"shadow_score"`

	// Timestamp is when the comparison was made.
	Timestamp time.Time `json:"timestamp"`
}
