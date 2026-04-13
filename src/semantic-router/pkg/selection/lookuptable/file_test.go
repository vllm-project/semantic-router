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

package lookuptable_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func newTempFileStorage(t *testing.T) (*lookuptable.FileStorage, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "lookup_tables.json")
	s, err := lookuptable.NewFileStorage(path)
	if err != nil {
		t.Fatalf("NewFileStorage: %v", err)
	}
	return s, path
}

func TestFileStorage_SaveAndLoad(t *testing.T) {
	s, path := newTempFileStorage(t)
	defer s.Close()

	key := lookuptable.QualityGapKey("coding", "gpt-4", "claude-3")
	entry := lookuptable.Entry{Value: 0.12, Source: lookuptable.SourceReplayDerived, UpdatedAt: time.Now(), SampleCount: 50}
	if err := s.Set(key, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Create a fresh storage instance and load.
	s2, err := lookuptable.NewFileStorage(path)
	if err != nil {
		t.Fatalf("NewFileStorage (2nd): %v", err)
	}
	defer s2.Close()
	if err := s2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	got, ok := s2.Get(key)
	if !ok {
		t.Fatal("entry not found after Load")
	}
	if got.Value != entry.Value {
		t.Errorf("loaded Value = %v, want %v", got.Value, entry.Value)
	}
	if got.SampleCount != entry.SampleCount {
		t.Errorf("loaded SampleCount = %d, want %d", got.SampleCount, entry.SampleCount)
	}
}

func TestFileStorage_NonExistentFile(t *testing.T) {
	s, _ := newTempFileStorage(t)
	defer s.Close()
	// File does not exist yet; Load should succeed with empty state.
	if err := s.Load(); err != nil {
		t.Errorf("Load on non-existent file should return nil, got: %v", err)
	}
	if len(s.All()) != 0 {
		t.Error("expected empty storage for non-existent file")
	}
}

func TestFileStorage_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")
	// Write an empty file.
	if err := os.WriteFile(path, []byte{}, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	s, err := lookuptable.NewFileStorage(path)
	if err != nil {
		t.Fatalf("NewFileStorage: %v", err)
	}
	defer s.Close()

	if err := s.Load(); err != nil {
		t.Errorf("Load on empty file should return nil, got: %v", err)
	}
}

func TestFileStorage_CorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.json")
	if err := os.WriteFile(path, []byte("not json {{{"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	s, err := lookuptable.NewFileStorage(path)
	if err != nil {
		t.Fatalf("NewFileStorage: %v", err)
	}
	defer s.Close()

	if err := s.Load(); err == nil {
		t.Error("Load on corrupt file should return an error")
	}

	// A .corrupted backup should have been created.
	if _, err := os.Stat(path + ".corrupted"); os.IsNotExist(err) {
		t.Error("expected .corrupted backup file to exist")
	}
}

func TestFileStorage_AtomicWrite(t *testing.T) {
	s, path := newTempFileStorage(t)
	defer s.Close()

	_ = s.Set(lookuptable.HandoffPenaltyKey("a", "b"), lookuptable.Entry{Value: 0.05})
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// The .tmp file should have been cleaned up.
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error("temp file should not exist after successful Save")
	}

	// The main file should exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("storage file should exist after Save")
	}
}

func TestFileStorage_StartAutoSave_IdempotentNoPanic(t *testing.T) {
	s, _ := newTempFileStorage(t)
	// Calling StartAutoSave twice must not panic (second call is a no-op).
	s.StartAutoSave(50 * time.Millisecond)
	s.StartAutoSave(50 * time.Millisecond) // must not panic
	s.Close()
}

func TestFileStorage_AutoSave(t *testing.T) {
	s, path := newTempFileStorage(t)

	_ = s.Set(lookuptable.RemainingTurnPriorKey("support"), lookuptable.Entry{Value: 3.0})
	s.StartAutoSave(50 * time.Millisecond)

	// Wait for at least one auto-save tick.
	time.Sleep(200 * time.Millisecond)
	s.Close()

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("file should exist after auto-save")
	}
}
