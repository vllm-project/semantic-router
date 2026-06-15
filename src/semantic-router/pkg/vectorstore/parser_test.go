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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ExtractText", func() {
	Context("with .txt files", func() {
		It("should return text content as-is", func() {
			content := []byte("Hello, world!\nSecond line.")
			result, err := ExtractText(content, "doc.txt")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal("Hello, world!\nSecond line."))
		})

		It("should handle empty content", func() {
			result, err := ExtractText([]byte(""), "empty.txt")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(""))
		})
	})

	Context("with .md files", func() {
		It("should return markdown content as-is", func() {
			content := []byte("# Title\n\nSome **bold** text.")
			result, err := ExtractText(content, "readme.md")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal("# Title\n\nSome **bold** text."))
		})
	})

	Context("with .json files", func() {
		It("should pretty-print valid JSON", func() {
			content := []byte(`{"name":"test","value":42}`)
			result, err := ExtractText(content, "data.json")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("\"name\": \"test\""))
			Expect(result).To(ContainSubstring("\"value\": 42"))
		})

		It("should handle JSON arrays", func() {
			content := []byte(`[1,2,3]`)
			result, err := ExtractText(content, "array.json")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("1"))
			Expect(result).To(ContainSubstring("2"))
			Expect(result).To(ContainSubstring("3"))
		})

		It("should return error for invalid JSON", func() {
			content := []byte(`{invalid json}`)
			_, err := ExtractText(content, "bad.json")

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to parse JSON"))
		})
	})

	Context("with .csv files", func() {
		It("should convert CSV to readable text", func() {
			content := []byte("Name,Age,City\nAlice,30,NYC\nBob,25,LA")
			result, err := ExtractText(content, "data.csv")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("Row 1:"))
			Expect(result).To(ContainSubstring("Name: Alice"))
			Expect(result).To(ContainSubstring("Age: 30"))
			Expect(result).To(ContainSubstring("City: NYC"))
			Expect(result).To(ContainSubstring("Row 2:"))
			Expect(result).To(ContainSubstring("Name: Bob"))
		})

		It("should handle empty CSV", func() {
			content := []byte("")
			result, err := ExtractText(content, "empty.csv")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(""))
		})

		It("should handle header-only CSV", func() {
			content := []byte("Name,Age,City\n")
			result, err := ExtractText(content, "header.csv")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(""))
		})
	})

	Context("with .html files", func() {
		It("should strip HTML tags and extract text", func() {
			content := []byte("<html><body><h1>Title</h1><p>Hello world</p></body></html>")
			result, err := ExtractText(content, "page.html")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("Title"))
			Expect(result).To(ContainSubstring("Hello world"))
			Expect(result).NotTo(ContainSubstring("<h1>"))
			Expect(result).NotTo(ContainSubstring("<p>"))
		})

		It("should handle .htm extension", func() {
			content := []byte("<p>Test content</p>")
			result, err := ExtractText(content, "page.htm")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("Test content"))
		})

		It("should skip script and style content", func() {
			content := []byte(`<html>
				<head><style>body { color: red; }</style></head>
				<body>
					<script>alert('hello');</script>
					<p>Visible text</p>
				</body>
			</html>`)
			result, err := ExtractText(content, "page.html")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("Visible text"))
			Expect(result).NotTo(ContainSubstring("alert"))
			Expect(result).NotTo(ContainSubstring("color: red"))
		})

		It("should handle nested elements", func() {
			content := []byte("<div><p>Nested <strong>bold</strong> text</p></div>")
			result, err := ExtractText(content, "nested.html")

			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(ContainSubstring("Nested"))
			Expect(result).To(ContainSubstring("bold"))
			Expect(result).To(ContainSubstring("text"))
		})
	})

	Context("with unsupported formats", func() {
		It("should return error for .pdf", func() {
			_, err := ExtractText([]byte("content"), "doc.pdf")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("unsupported file format"))
		})

		It("should return error for .docx", func() {
			_, err := ExtractText([]byte("content"), "doc.docx")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("unsupported file format"))
		})

		It("should return error for files with no extension", func() {
			_, err := ExtractText([]byte("content"), "noext")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("unsupported file format"))
		})
	})
})
