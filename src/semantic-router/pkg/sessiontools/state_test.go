package sessiontools

import (
	"testing"
	"time"
)

func validState() State {
	now := time.Now()
	return State{
		SchemaVersion:         SchemaVersion,
		Revision:              1,
		PolicyFingerprint:     "policy-fp",
		CatalogFingerprint:    "catalog-fp",
		CapabilityFingerprint: "capability-fp",
		Tools: []ToolState{
			{Name: "search", DefinitionFingerprint: "def-fp-1", FirstSeenTurn: 0},
			{Name: "lookup", DefinitionFingerprint: "def-fp-2", Pinned: true, FirstSeenTurn: 1},
		},
		CreatedAt:  now.Add(-time.Hour),
		LastSeenAt: now,
		ExpiresAt:  now.Add(time.Hour),
	}
}

func TestState_Validate_ValidStateOK(t *testing.T) {
	if err := validState().Validate(16, 16384); err != nil {
		t.Fatal(err)
	}
}

func TestState_Validate_WrongSchemaVersion_Err(t *testing.T) {
	s := validState()
	s.SchemaVersion = SchemaVersion + 1
	if err := s.Validate(16, 16384); err == nil {
		t.Fatal("expected error for unsupported schema_version")
	}
}

func TestState_Validate_ZeroSchemaVersion_Err(t *testing.T) {
	s := validState()
	s.SchemaVersion = 0
	if err := s.Validate(16, 16384); err == nil {
		t.Fatal("expected error for schema_version 0")
	}
}

func TestState_Validate_ZeroTimestamps_Err(t *testing.T) {
	cases := map[string]func(*State){
		"created_at zero":   func(s *State) { s.CreatedAt = time.Time{} },
		"last_seen_at zero": func(s *State) { s.LastSeenAt = time.Time{} },
		"expires_at zero":   func(s *State) { s.ExpiresAt = time.Time{} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			s := validState()
			mutate(&s)
			if err := s.Validate(16, 16384); err == nil {
				t.Fatalf("expected error for %s", name)
			}
		})
	}
}

func TestState_Validate_TimestampOrdering_Err(t *testing.T) {
	now := time.Now()
	t.Run("created_at after last_seen_at", func(t *testing.T) {
		s := validState()
		s.CreatedAt = now.Add(time.Minute)
		s.LastSeenAt = now
		if err := s.Validate(16, 16384); err == nil {
			t.Fatal("expected error: created_at after last_seen_at")
		}
	})
	t.Run("last_seen_at after expires_at", func(t *testing.T) {
		s := validState()
		s.LastSeenAt = now.Add(time.Hour)
		s.ExpiresAt = now
		if err := s.Validate(16, 16384); err == nil {
			t.Fatal("expected error: last_seen_at after expires_at")
		}
	})
	t.Run("all equal is allowed", func(t *testing.T) {
		s := validState()
		s.CreatedAt, s.LastSeenAt, s.ExpiresAt = now, now, now
		if err := s.Validate(16, 16384); err != nil {
			t.Fatal(err)
		}
	})
}

func TestState_Validate_ToolsExceedMaxTools_Err(t *testing.T) {
	s := validState() // 2 tools
	if err := s.Validate(1, 16384); err == nil {
		t.Fatal("expected error: tool count exceeds maxTools")
	}
	if err := s.Validate(2, 16384); err != nil {
		t.Fatal(err)
	}
}

func TestState_Validate_MaxToolsZeroMeansUnbounded(t *testing.T) {
	s := validState()
	if err := s.Validate(0, 16384); err != nil {
		t.Fatal(err)
	}
}

func TestState_Validate_DuplicateToolNames_Err(t *testing.T) {
	s := validState()
	s.Tools = []ToolState{
		{Name: "search", DefinitionFingerprint: "fp-1"},
		{Name: "search", DefinitionFingerprint: "fp-2"},
	}
	if err := s.Validate(16, 16384); err == nil {
		t.Fatal("expected error: duplicate tool name")
	}
}

func TestState_Validate_ToolFieldRequirements_Err(t *testing.T) {
	cases := map[string]ToolState{
		"empty name":                   {Name: "", DefinitionFingerprint: "fp"},
		"empty definition_fingerprint": {Name: "search", DefinitionFingerprint: ""},
		"negative first_seen_turn":     {Name: "search", DefinitionFingerprint: "fp", FirstSeenTurn: -1},
	}
	for name, tool := range cases {
		t.Run(name, func(t *testing.T) {
			s := validState()
			s.Tools = []ToolState{tool}
			if err := s.Validate(16, 16384); err == nil {
				t.Fatalf("expected error for %s", name)
			}
		})
	}
}

func TestState_Validate_EncodedSizeExceedsBound_Err(t *testing.T) {
	s := validState()
	if err := s.Validate(16, 1); err == nil {
		t.Fatal("expected error: encoded size exceeds a 1-byte bound")
	}
}

func TestState_Validate_MaxStateBytesZeroMeansUnbounded(t *testing.T) {
	s := validState()
	if err := s.Validate(16, 0); err != nil {
		t.Fatal(err)
	}
}

func TestState_Clone_DeepCopyIndependence(t *testing.T) {
	original := validState()
	cloned := original.Clone()

	if len(cloned.Tools) != len(original.Tools) {
		t.Fatalf("clone has %d tools, want %d", len(cloned.Tools), len(original.Tools))
	}
	for i := range cloned.Tools {
		if cloned.Tools[i] != original.Tools[i] {
			t.Fatalf("clone diverged in value at index %d: %+v vs %+v", i, cloned.Tools[i], original.Tools[i])
		}
	}

	// Mutating the clone's slice must not affect the original.
	cloned.Tools[0].Name = "mutated"
	cloned.Tools = append(cloned.Tools, ToolState{Name: "extra", DefinitionFingerprint: "fp"})
	if original.Tools[0].Name == "mutated" {
		t.Fatal("mutating the clone's tool leaked into the original")
	}
	if len(original.Tools) != 2 {
		t.Fatal("appending to the clone's Tools slice leaked into the original")
	}
}

func TestState_Clone_NilToolsStaysNil(t *testing.T) {
	s := validState()
	s.Tools = nil
	cloned := s.Clone()
	if cloned.Tools != nil {
		t.Fatalf("cloning a nil Tools slice should stay nil, got %+v", cloned.Tools)
	}
}
