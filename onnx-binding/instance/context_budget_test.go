//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Exercise the real loader with artifact-level fixed padding and actual
// postprocessor tokens; raw-token budgeting alone misses both conditions.
func specialTokenFixture(t *testing.T, kind string) Options {
	t.Helper()
	directory := t.TempDir()
	for _, name := range []string{"model.onnx", "config.json", "tokenizer.json"} {
		data, readErr := os.ReadFile(filepath.Join("testdata", kind, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if name == "tokenizer.json" {
			var tokenizer map[string]any
			if decodeErr := json.Unmarshal(data, &tokenizer); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			tokenizer["post_processor"] = map[string]any{"type": "BertProcessing", "sep": []any{"test", 4}, "cls": []any{"[UNK]", 0}}
			tokenizer["padding"] = map[string]any{"strategy": map[string]any{"Fixed": 4096}, "direction": "Right", "pad_to_multiple_of": nil, "pad_id": 0, "pad_type_id": 0, "pad_token": "[UNK]"}
			var encodeErr error
			data, encodeErr = json.Marshal(tokenizer)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
		}
		if writeErr := os.WriteFile(filepath.Join(directory, name), data, 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	return Options{ModelPath: directory, Provider: "cpu", IntraThreads: 1, MaxInputTokens: 4}
}

func TestOwnedClassifierRetainsUpstreamSpecialTokenAndPaddingSafeguards(t *testing.T) {
	options := specialTokenFixture(t, "sequence")
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	result, err := model.Classify("hello")
	if err != nil {
		t.Fatal(err)
	}
	// Actual IDs are [CLS=0, hello=1, SEP=4]. Fixed artifact padding must
	// not lower the real graph's mean logit, even below the task budget.
	want := 1 / (1 + math.Exp(-2*5.0/3.0))
	if math.Abs(float64(result.Confidence)-want) > 1e-6 || result.Input.OriginalTokens != 3 || result.Input.ProcessedTokens != 3 || result.Input.Truncated {
		t.Fatalf("special-token/padding semantics lost: %+v", result)
	}
	options.MaxInputTokens = 1
	if invalid, loadErr := LoadSequenceClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("budget smaller than actual special tokens was accepted")
	} else if !strings.Contains(loadErr.Error(), "requires 2 special tokens") {
		t.Fatalf("wrong preparation error: %v", loadErr)
	}
	options.MaxInputTokens = 513
	if invalid, loadErr := LoadSequenceClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("owned Core classification cap exceeded 512")
	}
	options.MaxInputTokens, options.Overflow = 2, "truncate_right"
	truncated, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer truncated.Close()
	trimmed, err := truncated.Classify("hello")
	if err != nil {
		t.Fatal(err)
	}
	if trimmed.Input.OriginalTokens != 3 || trimmed.Input.ProcessedTokens != 2 || !trimmed.Input.Truncated {
		t.Fatalf("special tokens escaped usage budget: %+v", trimmed)
	}
}

func TestOwnedTokenClassifierRetainsEffectiveBudgetAndBIOOffsets(t *testing.T) {
	options := specialTokenFixture(t, "token")
	model, err := LoadTokenClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	result, err := model.Detect("秘密")
	if err != nil {
		t.Fatal(err)
	}
	if result.Input.OriginalTokens != 3 || result.Input.ProcessedTokens != 3 || len(result.Spans) != 1 {
		t.Fatalf("invalid token output: %+v", result)
	}
	span := result.Spans[0]
	if span.Text != "秘密" || span.Start != 0 || span.End != len("秘密") || span.EntityType != "SECRET" {
		t.Fatalf("special tokens affected BIO byte spans: %+v", span)
	}
	options.MaxInputTokens = 1
	if invalid, loadErr := LoadTokenClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("token loader bypassed special-token minimum")
	}
}

func TestMIGraphXOwnedClassifierUsesMaskedFixedExecutionBudget(t *testing.T) {
	if os.Getenv("ORT_TEST_MIGRAPHX") != "1" {
		t.Skip("requires a real MIGraphX runtime and device")
	}
	for _, kind := range []string{"sequence", "token"} {
		t.Run(kind, func(t *testing.T) {
			options := specialTokenFixture(t, kind)
			options.Provider, options.MaxInputTokens = "migraphx", 8
			options.ProfilePrefix = filepath.Join(t.TempDir(), "fixed-budget")
			configPath := filepath.Join(options.ModelPath, "config.json")
			data, readErr := os.ReadFile(configPath)
			if readErr != nil {
				t.Fatal(readErr)
			}
			var config map[string]any
			if decodeErr := json.Unmarshal(data, &config); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			// A nonzero pad ID catches an accidentally unmasked execution tail.
			config["pad_token_id"] = 99
			data, encodeErr := json.Marshal(config)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			if writeErr := os.WriteFile(configPath, data, 0o600); writeErr != nil {
				t.Fatal(writeErr)
			}
			if kind == "sequence" {
				model, loadErr := LoadSequenceClassifier(options)
				if loadErr != nil {
					t.Fatal(loadErr)
				}
				defer model.Close()
				for _, input := range []struct {
					text   string
					sum    float64
					tokens int
				}{{"hello", 5, 3}, {"hello world", 7, 4}} {
					result, inferErr := model.Classify(input.text)
					if inferErr != nil {
						t.Fatal(inferErr)
					}
					// This existing graph has an unmasked ReduceMean denominator.
					// Its actual padded output changes: preserve that evidence.
					want := 1 / (1 + math.Exp(-2*input.sum/8))
					if math.Abs(float64(result.Confidence)-want) > 1e-6 || result.Input.OriginalTokens != input.tokens || result.Input.ProcessedTokens != input.tokens || result.Input.Truncated {
						t.Fatalf("wrong padded tensor or real-token usage: %+v", result)
					}
				}
				paths, profileErr := model.FinishProfiling()
				if profileErr != nil {
					t.Fatal(profileErr)
				}
				assertStrictGPUProfiles(t, paths)
				return
			}
			model, loadErr := LoadTokenClassifier(options)
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			defer model.Close()
			result, inferErr := model.Detect("秘密")
			if inferErr != nil {
				t.Fatal(inferErr)
			}
			if len(result.Spans) != 1 || result.Input.OriginalTokens != 3 || result.Input.ProcessedTokens != 3 || result.Input.Truncated {
				t.Fatalf("padded token rows changed actual usage/entities: %+v", result)
			}
			span := result.Spans[0]
			if span.Text != "秘密" || span.Start != 0 || span.End != len("秘密") || span.EntityType != "SECRET" || math.Abs(float64(span.Confidence)-1/(1+math.Exp(-6))) > 1e-6 {
				t.Fatalf("padded token rows changed BIO span or confidence: %+v", span)
			}
			paths, profileErr := model.FinishProfiling()
			if profileErr != nil {
				t.Fatal(profileErr)
			}
			assertStrictGPUProfiles(t, paths)
		})
	}
}
