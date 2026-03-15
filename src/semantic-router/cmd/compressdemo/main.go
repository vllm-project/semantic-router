// cmd/compressdemo/main.go — CLI harness for the prompt compression demo.
// Reads JSON lines from stdin, each {"id": "...", "text": "...", "max_tokens": N},
// runs the real compressor, and writes JSON lines to stdout with per-sentence
// scores and composite results.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

type InputRecord struct {
	ID        string `json:"id"`
	Text      string `json:"text"`
	MaxTokens int    `json:"max_tokens"`
}

type SentenceOutput struct {
	Index     int     `json:"idx"`
	Text      string  `json:"text"`
	Tokens    int     `json:"tokens"`
	TextRank  float64 `json:"textrank"`
	Position  float64 `json:"position"`
	TFIDF     float64 `json:"tfidf"`
	Novelty   float64 `json:"novelty"`
	Composite float64 `json:"composite"`
	Kept      bool    `json:"kept"`
}

type OutputRecord struct {
	ID               string           `json:"id"`
	OriginalTokens   int              `json:"original_tokens"`
	CompressedTokens int              `json:"compressed_tokens"`
	Ratio            float64          `json:"ratio"`
	Compressed       string           `json:"compressed"`
	Sentences        []SentenceOutput `json:"sentences"`
}

func main() {
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)

	for {
		var rec InputRecord
		if err := dec.Decode(&rec); err != nil {
			break
		}

		cfg := promptcompression.DefaultConfig(rec.MaxTokens)
		result := promptcompression.Compress(rec.Text, cfg)

		keptSet := make(map[int]bool, len(result.KeptIndices))
		for _, idx := range result.KeptIndices {
			keptSet[idx] = true
		}

		sentences := make([]SentenceOutput, len(result.SentenceScores))
		for i, ss := range result.SentenceScores {
			sentences[i] = SentenceOutput{
				Index:     ss.Index,
				Text:      ss.Text,
				Tokens:    ss.Tokens,
				TextRank:  ss.TextRank,
				Position:  ss.Position,
				TFIDF:     ss.TFIDF,
				Novelty:   ss.Novelty,
				Composite: ss.Composite,
				Kept:      keptSet[ss.Index],
			}
		}

		out := OutputRecord{
			ID:               rec.ID,
			OriginalTokens:   result.OriginalTokens,
			CompressedTokens: result.CompressedTokens,
			Ratio:            result.Ratio,
			Compressed:       result.Compressed,
			Sentences:        sentences,
		}
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
		}
	}
}
