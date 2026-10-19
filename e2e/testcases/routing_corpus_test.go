package testcases

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRoutingCorpusLoadersRejectInvalidData(t *testing.T) {
	for _, loader := range []struct {
		name string
		load func(string) (int, error)
	}{
		{
			name: "model selection",
			load: func(path string) (int, error) {
				cases, err := loadModelSelectionCases(path)
				return len(cases), err
			},
		},
		{
			name: "signal routing",
			load: func(path string) (int, error) {
				cases, err := loadSignalRoutingCases(path)
				return len(cases), err
			},
		},
	} {
		t.Run(loader.name, func(t *testing.T) {
			for _, corpus := range []struct {
				name    string
				body    string
				missing bool
				wantErr bool
			}{
				{name: "empty array", body: "[]", wantErr: true},
				{name: "null", body: "null", wantErr: true},
				{name: "malformed", body: "[", wantErr: true},
				{name: "missing file", missing: true, wantErr: true},
				{name: "one probe", body: `[{"query":"Explain a hash table"}]`},
			} {
				t.Run(corpus.name, func(t *testing.T) {
					path := filepath.Join(t.TempDir(), "cases.json")
					if !corpus.missing {
						if writeErr := os.WriteFile(path, []byte(corpus.body), 0o600); writeErr != nil {
							t.Fatal(writeErr)
						}
					}
					count, err := loader.load(path)
					if (err != nil) != corpus.wantErr {
						t.Fatalf("load returned %d probes and error %v, want error=%v", count, err, corpus.wantErr)
					}
					if !corpus.wantErr && count != 1 {
						t.Fatalf("loaded %d probes, want 1", count)
					}
				})
			}
		})
	}
}

func TestRoutingCorporaAreNonEmpty(t *testing.T) {
	if _, err := loadModelSelectionCases("testdata/model_selection_cases.json"); err != nil {
		t.Fatal(err)
	}
	for _, signal := range []string{
		"action", "complexity", "context", "conversation", "event",
		"language", "preference", "reask", "structure",
	} {
		t.Run(signal, func(t *testing.T) {
			if _, err := loadSignalRoutingCases("testdata/" + signal + "_routing_cases.json"); err != nil {
				t.Fatal(err)
			}
		})
	}
}
