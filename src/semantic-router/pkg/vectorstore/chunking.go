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

package vectorstore

import (
	"strings"
)

// Default chunking parameters (character-based approximation of tokens).
const (
	DefaultMaxChunkSize = 800
	DefaultChunkOverlap = 400
)

// ChunkText splits text into chunks according to the given strategy.
// If strategy is nil, "auto" strategy is used.
func ChunkText(text string, strategy *ChunkingStrategy) []TextChunk {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	if strategy == nil || strategy.Type == "" || strategy.Type == "auto" {
		return chunkAuto(text)
	}

	if strategy.Type == "static" {
		maxSize := DefaultMaxChunkSize
		overlap := DefaultChunkOverlap
		if strategy.Static != nil {
			if strategy.Static.MaxChunkSizeTokens > 0 {
				maxSize = strategy.Static.MaxChunkSizeTokens
			}
			if strategy.Static.ChunkOverlapTokens > 0 {
				overlap = strategy.Static.ChunkOverlapTokens
			}
		}
		return chunkStatic(text, maxSize, overlap)
	}

	// Unknown strategy: fall back to auto.
	return chunkAuto(text)
}

// chunkAuto splits text by paragraphs (double newlines). Consecutive blank
// lines are collapsed. Markdown headings (lines starting with #) also act
// as paragraph boundaries.
func chunkAuto(text string) []TextChunk {
	// Split on double newlines.
	paragraphs := strings.Split(text, "\n\n")

	var chunks []TextChunk
	idx := 0
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}

		// Further split on markdown headings within a paragraph block.
		subParts := splitOnHeadings(p)
		for _, part := range subParts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			chunks = append(chunks, TextChunk{
				Content:    part,
				ChunkIndex: idx,
			})
			idx++
		}
	}

	return chunks
}

// splitOnHeadings splits a block of text whenever a line starts with one
// or more '#' characters (markdown heading). The heading line is kept as
// part of the following chunk.
func splitOnHeadings(block string) []string {
	lines := strings.Split(block, "\n")
	var parts []string
	var current []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") && len(current) > 0 {
			parts = append(parts, strings.Join(current, "\n"))
			current = nil
		}
		current = append(current, line)
	}
	if len(current) > 0 {
		parts = append(parts, strings.Join(current, "\n"))
	}

	return parts
}

// chunkStatic splits text into fixed-size chunks with overlap.
// maxSize is the maximum number of characters per chunk.
// overlap is the number of characters that overlap between consecutive chunks.
func chunkStatic(text string, maxSize, overlap int) []TextChunk {
	if maxSize <= 0 {
		maxSize = DefaultMaxChunkSize
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= maxSize {
		overlap = maxSize / 2
	}

	runes := []rune(text)
	total := len(runes)
	if total == 0 {
		return nil
	}

	var chunks []TextChunk
	idx := 0
	start := 0

	for start < total {
		end := start + maxSize
		if end > total {
			end = total
		}

		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, TextChunk{
				Content:    chunk,
				ChunkIndex: idx,
			})
			idx++
		}

		// Advance by (maxSize - overlap) to create the overlap region.
		step := maxSize - overlap
		if step <= 0 {
			step = 1
		}
		start += step
	}

	return chunks
}
