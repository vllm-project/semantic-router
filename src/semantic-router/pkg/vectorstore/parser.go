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
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"golang.org/x/net/html"
)

// ExtractText extracts plain text from file content based on the file extension.
// Supported extensions: .txt, .md, .json, .csv, .html
func ExtractText(content []byte, filename string) (string, error) {
	ext := strings.ToLower(filepath.Ext(filename))

	switch ext {
	case ".txt", ".md":
		return string(content), nil
	case ".json":
		return extractJSON(content)
	case ".csv":
		return extractCSV(content)
	case ".html", ".htm":
		return extractHTML(content)
	default:
		return "", fmt.Errorf("unsupported file format: %s", ext)
	}
}

// extractJSON pretty-prints JSON for readability.
func extractJSON(content []byte) (string, error) {
	var data interface{}
	if err := json.Unmarshal(content, &data); err != nil {
		return "", fmt.Errorf("failed to parse JSON: %w", err)
	}

	pretty, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to format JSON: %w", err)
	}

	return string(pretty), nil
}

// extractCSV converts CSV rows to a readable text representation.
func extractCSV(content []byte) (string, error) {
	reader := csv.NewReader(bytes.NewReader(content))

	// Read header row.
	headers, err := reader.Read()
	if err != nil {
		if err == io.EOF {
			return "", nil
		}
		return "", fmt.Errorf("failed to read CSV header: %w", err)
	}

	var b strings.Builder
	rowNum := 0

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("failed to read CSV row %d: %w", rowNum+1, err)
		}

		rowNum++
		b.WriteString(fmt.Sprintf("Row %d:\n", rowNum))
		for i, val := range record {
			header := ""
			if i < len(headers) {
				header = headers[i]
			} else {
				header = fmt.Sprintf("Column%d", i+1)
			}
			b.WriteString(fmt.Sprintf("  %s: %s\n", header, val))
		}
	}

	return b.String(), nil
}

// extractHTML strips HTML tags and returns the text content.
func extractHTML(content []byte) (string, error) {
	tokenizer := html.NewTokenizer(bytes.NewReader(content))

	var b strings.Builder
	var skipContent bool

	for {
		tt := tokenizer.Next()
		switch tt {
		case html.ErrorToken:
			err := tokenizer.Err()
			if errors.Is(err, io.EOF) {
				return strings.TrimSpace(b.String()), nil
			}
			return "", fmt.Errorf("failed to parse HTML: %w", err)

		case html.StartTagToken:
			tn, _ := tokenizer.TagName()
			tag := string(tn)
			// Skip content inside script and style tags.
			if tag == "script" || tag == "style" {
				skipContent = true
			}

		case html.EndTagToken:
			tn, _ := tokenizer.TagName()
			tag := string(tn)
			if tag == "script" || tag == "style" {
				skipContent = false
			}
			// Add spacing after block-level elements.
			if isBlockElement(tag) {
				b.WriteString("\n")
			}

		case html.TextToken:
			if !skipContent {
				text := strings.TrimSpace(string(tokenizer.Text()))
				if text != "" {
					b.WriteString(text)
					b.WriteString(" ")
				}
			}
		}
	}
}

// isBlockElement returns true for HTML block-level elements that should
// produce line breaks in the extracted text.
func isBlockElement(tag string) bool {
	switch tag {
	case "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
		"li", "tr", "blockquote", "pre", "section", "article",
		"header", "footer", "nav", "aside":
		return true
	}
	return false
}
