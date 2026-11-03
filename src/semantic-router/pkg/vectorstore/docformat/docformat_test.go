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

package docformat

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		".txt":    ".txt",
		"txt":     ".txt",
		"TXT":     ".txt",
		" .Md ":   ".md",
		".HTML":   ".html",
		"":        "",
		"   ":     "",
		".tar.gz": ".tar.gz",
	}
	for format, want := range cases {
		assert.Equal(t, want, Normalize(format), "format %q", format)
	}
}

func TestSupported(t *testing.T) {
	for _, format := range []string{".txt", "txt", ".MD", ".json", ".csv", ".html", ".htm"} {
		assert.True(t, Supported(format), "format %q", format)
	}
	for _, format := range []string{".pdf", "pdf", ".docx", ".png", "", "  "} {
		assert.False(t, Supported(format), "format %q", format)
	}
}

func TestExtensions(t *testing.T) {
	assert.Equal(t, []string{".csv", ".htm", ".html", ".json", ".md", ".txt"}, Extensions())
	assert.Equal(t, ".csv, .htm, .html, .json, .md, .txt", List())
}
