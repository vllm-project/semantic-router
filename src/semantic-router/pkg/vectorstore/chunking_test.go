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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ChunkText", func() {
	Context("with auto strategy", func() {
		It("should split text on double newlines", func() {
			text := "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
			chunks := ChunkText(text, nil)

			Expect(chunks).To(HaveLen(3))
			Expect(chunks[0].Content).To(Equal("First paragraph."))
			Expect(chunks[0].ChunkIndex).To(Equal(0))
			Expect(chunks[1].Content).To(Equal("Second paragraph."))
			Expect(chunks[1].ChunkIndex).To(Equal(1))
			Expect(chunks[2].Content).To(Equal("Third paragraph."))
			Expect(chunks[2].ChunkIndex).To(Equal(2))
		})

		It("should handle explicit auto strategy", func() {
			text := "Para one.\n\nPara two."
			strategy := &ChunkingStrategy{Type: "auto"}
			chunks := ChunkText(text, strategy)

			Expect(chunks).To(HaveLen(2))
			Expect(chunks[0].Content).To(Equal("Para one."))
			Expect(chunks[1].Content).To(Equal("Para two."))
		})

		It("should collapse multiple blank lines", func() {
			text := "First.\n\n\n\n\nSecond."
			chunks := ChunkText(text, nil)

			Expect(chunks).To(HaveLen(2))
			Expect(chunks[0].Content).To(Equal("First."))
			Expect(chunks[1].Content).To(Equal("Second."))
		})

		It("should split on markdown headings", func() {
			text := "Introduction text.\n# Heading One\nContent under heading one."
			chunks := ChunkText(text, nil)

			Expect(chunks).To(HaveLen(2))
			Expect(chunks[0].Content).To(Equal("Introduction text."))
			Expect(chunks[1].Content).To(ContainSubstring("# Heading One"))
			Expect(chunks[1].Content).To(ContainSubstring("Content under heading one."))
		})

		It("should return nil for empty text", func() {
			chunks := ChunkText("", nil)
			Expect(chunks).To(BeNil())
		})

		It("should return nil for whitespace-only text", func() {
			chunks := ChunkText("   \n\n   \t  ", nil)
			Expect(chunks).To(BeNil())
		})

		It("should handle single line text", func() {
			text := "Just a single line."
			chunks := ChunkText(text, nil)

			Expect(chunks).To(HaveLen(1))
			Expect(chunks[0].Content).To(Equal("Just a single line."))
			Expect(chunks[0].ChunkIndex).To(Equal(0))
		})

		It("should trim whitespace from chunks", func() {
			text := "  First paragraph.  \n\n  Second paragraph.  "
			chunks := ChunkText(text, nil)

			Expect(chunks).To(HaveLen(2))
			Expect(chunks[0].Content).To(Equal("First paragraph."))
			Expect(chunks[1].Content).To(Equal("Second paragraph."))
		})
	})

	Context("with static strategy", func() {
		It("should split text into fixed-size chunks", func() {
			text := strings.Repeat("a", 100)
			strategy := &ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 30,
					ChunkOverlapTokens: 0,
				},
			}

			chunks := ChunkText(text, strategy)

			Expect(len(chunks)).To(BeNumerically(">=", 3))
			for _, c := range chunks {
				Expect(len(c.Content)).To(BeNumerically("<=", 30))
			}
		})

		It("should create overlap between chunks", func() {
			text := "0123456789abcdefghij"
			strategy := &ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 10,
					ChunkOverlapTokens: 5,
				},
			}

			chunks := ChunkText(text, strategy)

			Expect(len(chunks)).To(BeNumerically(">=", 2))
			// Verify overlap: the end of chunk 0 should appear at the start of chunk 1.
			if len(chunks) >= 2 {
				overlapContent := chunks[0].Content[5:]
				Expect(chunks[1].Content).To(HavePrefix(overlapContent))
			}
		})

		It("should use default values when config is nil", func() {
			text := strings.Repeat("x", 2000)
			strategy := &ChunkingStrategy{
				Type: "static",
			}

			chunks := ChunkText(text, strategy)
			Expect(len(chunks)).To(BeNumerically(">=", 2))
		})

		It("should handle text shorter than maxSize", func() {
			text := "Short text"
			strategy := &ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 100,
					ChunkOverlapTokens: 10,
				},
			}

			chunks := ChunkText(text, strategy)

			Expect(chunks).To(HaveLen(1))
			Expect(chunks[0].Content).To(Equal("Short text"))
		})

		It("should handle overlap >= maxSize by capping", func() {
			text := strings.Repeat("y", 50)
			strategy := &ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 10,
					ChunkOverlapTokens: 10, // equal to max, should be capped to 5
				},
			}

			chunks := ChunkText(text, strategy)
			Expect(len(chunks)).To(BeNumerically(">=", 2))
		})

		It("should produce sequential chunk indices", func() {
			text := strings.Repeat("z", 200)
			strategy := &ChunkingStrategy{
				Type: "static",
				Static: &StaticChunkConfig{
					MaxChunkSizeTokens: 30,
					ChunkOverlapTokens: 10,
				},
			}

			chunks := ChunkText(text, strategy)
			for i, c := range chunks {
				Expect(c.ChunkIndex).To(Equal(i))
			}
		})

		It("should return nil for empty text", func() {
			strategy := &ChunkingStrategy{Type: "static"}
			chunks := ChunkText("", strategy)
			Expect(chunks).To(BeNil())
		})
	})

	Context("with unknown strategy", func() {
		It("should fall back to auto", func() {
			text := "Paragraph one.\n\nParagraph two."
			strategy := &ChunkingStrategy{Type: "unknown_strategy"}
			chunks := ChunkText(text, strategy)

			Expect(chunks).To(HaveLen(2))
		})
	})
})
