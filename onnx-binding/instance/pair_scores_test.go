//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOwnedPairScorerUsesEncoderNormalizationContract(t *testing.T) {
	for _, variant := range []string{"declared", "legacy_false", "conflicting_contract"} {
		t.Run(variant, func(t *testing.T) {
			options := pairFixture()
			directory := t.TempDir()
			for _, name := range []string{"model.onnx", "config.json", "matryoshka_config.json", "tokenizer.json"} {
				data, readErr := os.ReadFile(filepath.Join(options.ModelPath, name))
				if readErr != nil {
					t.Fatal(readErr)
				}
				if writeErr := os.WriteFile(filepath.Join(directory, name), data, 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
			}
			options.ModelPath = directory
			read := func(name string) map[string]any {
				data, err := os.ReadFile(filepath.Join(directory, name))
				if err != nil {
					t.Fatal(err)
				}
				var result map[string]any
				if err := json.Unmarshal(data, &result); err != nil {
					t.Fatal(err)
				}
				return result
			}
			layout := read("matryoshka_config.json")
			delete(layout, "has_final_norm")
			representation := read("config.json")["representation_contract"].(map[string]any)
			layout["representation_contract"] = representation
			if variant == "legacy_false" {
				layout["has_final_norm"] = false
			}
			if variant == "conflicting_contract" {
				representation["final_normalization"] = "none"
			}
			data, err := json.Marshal(layout)
			if err != nil {
				t.Fatal(err)
			}
			if writeErr := os.WriteFile(filepath.Join(directory, "matryoshka_config.json"), data, 0o600); writeErr != nil {
				t.Fatal(writeErr)
			}
			model, err := LoadPairScorer(options, PairScorerSelection{})
			if variant != "declared" {
				if err == nil {
					_ = model.Close()
					t.Fatal("contradictory normalization accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer model.Close()
			result, err := model.ScorePairs([]TextPair{{Query: "hello world", Document: "秘密 hello"}})
			if err != nil || len(result.Scores) != 1 || result.Scores[0] != 15 {
				t.Fatalf("declared normalization changed pair result: %+v, %v", result, err)
			}
		})
	}
}

func pairFixture() Options {
	return Options{ModelPath: filepath.Join("testdata", "pair_scorer"), ModelFile: "model.onnx", Provider: "cpu", IntraThreads: 1}
}

func TestOwnedPairScorerUsesRealNativePairsAndRawLogits(t *testing.T) {
	options := pairFixture()
	options.MaxInputTokens = 7
	model, err := LoadPairScorer(options, PairScorerSelection{})
	if err != nil {
		t.Fatal(err)
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	pairs := []TextPair{{Query: "hello world", Document: "秘密 hello"}, {Query: "world", Document: "秘密"}}
	if _, err = model.ScorePairs(pairs); err == nil {
		t.Fatal("closed owner accepted pair scoring")
	}
	result, err := clone.ScorePairs(pairs)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 2 || result.Scores[0] != 15 || result.Scores[1] != 13 || len(result.Inputs) != 2 || result.Inputs[0].OriginalTokens != 7 || result.Inputs[0].ProcessedTokens != 7 || result.Inputs[0].Truncated {
		t.Fatalf("exact pair IDs or raw logits changed: %+v", result)
	}
	if _, err = clone.ScorePairs([]TextPair{{Query: string([]byte{0xff}), Document: "hello"}}); err == nil {
		t.Fatal("invalid UTF-8 was silently replaced by JSON encoding")
	}
	info, err := clone.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Task != "pair_scores" || info.PairScorer == nil || *info.PairScorer != (PairScorerSelection{Layer: 2, Dimension: 4}) || info.CompletedInferences != 2 {
		t.Fatalf("wrong actual head/session evidence: %+v", info)
	}
	if _, err = clone.ScorePairs([]TextPair{pairs[0], {Query: "hello world", Document: "秘密 hello world"}}); err == nil {
		t.Fatal("over-budget pair batch was accepted")
	}
	info, err = clone.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.CompletedInferences != 2 {
		t.Fatal("failed preparation still ran part of the pair batch")
	}
	if invalid, loadErr := LoadPairScorer(options, PairScorerSelection{Layer: 1, Dimension: 4}); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("wrong graph exit accepted")
	}
	options.Overflow = "truncate_right"
	if invalid, loadErr := LoadPairScorer(options, PairScorerSelection{}); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("pair template truncation accepted")
	}
}

func TestOwnedPairScorerProcessesAll32768TokensAndRejects32769(t *testing.T) {
	model, err := LoadPairScorer(pairFixture(), PairScorerSelection{})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	pair := TextPair{Query: "hello", Document: strings.Repeat("world ", 32764)}
	result, err := model.ScorePairs([]TextPair{pair})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 1 || result.Scores[0] != 65537 || result.Inputs[0].OriginalTokens != 32768 || result.Inputs[0].ProcessedTokens != 32768 || result.Inputs[0].Truncated {
		t.Fatalf("32K pair content was lost: %+v", result)
	}
	pair.Document += "world"
	if _, err = model.ScorePairs([]TextPair{pair}); err == nil {
		t.Fatal("32769-token pair accepted")
	}
}
