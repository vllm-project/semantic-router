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

// Package docformat names the document formats the vector store can ingest.
//
// It is the single list of formats that have a text extractor: the
// extraction itself, the upload route and the configuration validation all
// read it, so a configured format can never name a document the extractor
// cannot read.
package docformat

import (
	"maps"
	"slices"
	"strings"
)

// Extractor names the text extraction a document format uses.
type Extractor int

const (
	// PlainText keeps the document's bytes as they are.
	PlainText Extractor = iota
	// JSON pretty-prints the document.
	JSON
	// CSV renders the document's rows.
	CSV
	// HTML renders the document's visible text.
	HTML
)

// Formats maps every document extension with a text extractor to the
// extraction it uses. An extension that is absent has no text extractor.
var Formats = map[string]Extractor{
	".txt":  PlainText,
	".md":   PlainText,
	".json": JSON,
	".csv":  CSV,
	".html": HTML,
	".htm":  HTML,
}

// Extensions lists the extensions with a text extractor, sorted.
func Extensions() []string {
	return slices.Sorted(maps.Keys(Formats))
}

// Normalize returns the canonical extension of a format: lowercased and
// dot-prefixed, so "TXT", "txt" and ".txt" name the same format. An empty
// name stays empty.
func Normalize(format string) string {
	ext := strings.ToLower(strings.TrimSpace(format))
	if ext == "" {
		return ""
	}
	if !strings.HasPrefix(ext, ".") {
		ext = "." + ext
	}
	return ext
}

// Supported reports whether the format has a text extractor.
func Supported(format string) bool {
	_, ok := Formats[Normalize(format)]
	return ok
}

// List renders the formats with a text extractor for an error message.
func List() string {
	return strings.Join(Extensions(), ", ")
}
