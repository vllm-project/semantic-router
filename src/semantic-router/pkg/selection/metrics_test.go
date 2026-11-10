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
	"testing"
	"time"
)

func TestInitializeMetrics(t *testing.T) {
	InitializeMetrics()

	// Verify metric pointers are not nil
	if ModelSelectionTotal == nil {
		t.Error("ModelSelectionTotal should not be nil")
	}
	if ModelSelectionDuration == nil {
		t.Error("ModelSelectionDuration should not be nil")
	}
	if ModelSelectionScore == nil {
		t.Error("ModelSelectionScore should not be nil")
	}
	if ModelSelectionConfidence == nil {
		t.Error("ModelSelectionConfidence should not be nil")
	}
	if ModelSelectionHistory == nil {
		t.Error("ModelSelectionHistory should not be nil")
	}
	if ModelSelectionDependencyHealth == nil {
		t.Error("ModelSelectionDependencyHealth should not be nil")
	}
}

func TestRecordSelection(t *testing.T) {
	InitializeMetrics()

	// Should not panic
	RecordSelection("static", "tech", "llama3.2:3b", TierSupported, 0.85)
	RecordSelection("multi_factor", "finance", "phi4", TierSupported, 0.72)
	RecordSelection("systemone", "general", "gemma3:27b", TierSupported, 0.91)
}

func TestRecordSelectionWithTier(t *testing.T) {
	InitializeMetrics()

	// Record a selection with tier - should not panic
	RecordSelectionWithTier(MethodStatic, "test-model", "test-decision", TierSupported, 0.8, 0.9, time.Millisecond)
	RecordSelectionWithTier(MethodSystemOne, "test-model", "test-decision", TierSupported, 0.7, 0.8, time.Millisecond)
}

func TestRecordDependencyHealth(t *testing.T) {
	InitializeMetrics()

	// Should not panic
	RecordDependencyHealth("systemone", "Decision model runtime", "external_service", false)
	RecordDependencyHealth("static", "", "", true)
}
