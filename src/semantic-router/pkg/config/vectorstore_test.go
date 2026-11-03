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

package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVectorStoreSupportedFormatsValidation(t *testing.T) {
	cases := []struct {
		name    string
		formats []string
		wantErr string
	}{
		{
			name:    "the formats the extractor reads are accepted",
			formats: []string{".txt", ".md", ".json", ".csv", ".html", ".htm"},
		},
		{
			name:    "a format without its dot is accepted",
			formats: []string{"txt", "md"},
		},
		{
			name:    "an upper case format is accepted",
			formats: []string{".TXT", ".HTML"},
		},
		{
			name:    "no formats are accepted",
			formats: nil,
		},
		{
			name:    "a format with no text extractor is refused",
			formats: []string{".txt", ".pdf"},
			wantErr: "vector_store.supported_formats lists '.pdf', which has no text extractor",
		},
		{
			name:    "a format without its dot and with no text extractor is refused",
			formats: []string{"pdf"},
			wantErr: "vector_store.supported_formats lists 'pdf', which has no text extractor",
		},
		{
			name:    "an upper case format with no text extractor is refused",
			formats: []string{".PDF"},
			wantErr: "vector_store.supported_formats lists '.PDF', which has no text extractor",
		},
		{
			name:    "an empty format is refused",
			formats: []string{".txt", "  "},
			wantErr: "vector_store.supported_formats must not list an empty format",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &VectorStoreConfig{
				Enabled:          true,
				BackendType:      "memory",
				SupportedFormats: tc.formats,
			}
			err := cfg.Validate()
			if tc.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestVectorStoreSupportedFormatsAreIgnoredWhenDisabled(t *testing.T) {
	cfg := &VectorStoreConfig{
		Enabled:          false,
		SupportedFormats: []string{".pdf"},
	}
	assert.NoError(t, cfg.Validate())
}

func TestVectorStoreSupportedFormatErrorNamesTheReadableFormats(t *testing.T) {
	cfg := &VectorStoreConfig{
		Enabled:          true,
		BackendType:      "memory",
		SupportedFormats: []string{".pdf"},
	}
	err := cfg.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), ".csv, .htm, .html, .json, .md, .txt")
}

// The shipped configuration must only name formats the extractor reads, or
// the router accepts an upload it can never index.
func TestReferenceConfigVectorStoreFormatsCanBeIngested(t *testing.T) {
	root := repoRootFromTestFile(t)
	cfg, err := Parse(root + "/config/config.yaml")
	require.NoError(t, err)
	require.NotNil(t, cfg.VectorStore)
	assert.NoError(t, cfg.VectorStore.Validate())
}

// The formats are checked wherever the configuration is loaded or replaced,
// so a configuration that names a format nobody can read never reaches the
// upload route.
func TestConfigValidationRejectsAFormatWithNoTextExtractor(t *testing.T) {
	cfg := &RouterConfig{
		VectorStore: &VectorStoreConfig{
			Enabled:          true,
			BackendType:      "memory",
			SupportedFormats: []string{".txt", ".pdf"},
		},
	}
	err := validateConfigStructure(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vector_store.supported_formats lists '.pdf'")
}

func TestConfigValidationAcceptsTheFormatsTheExtractorReads(t *testing.T) {
	cfg := &RouterConfig{
		VectorStore: &VectorStoreConfig{
			Enabled:          true,
			BackendType:      "memory",
			SupportedFormats: []string{".txt", ".md", ".json", ".csv", ".html", ".htm"},
		},
	}
	assert.NoError(t, validateConfigStructure(cfg))
}

func TestConfigValidationIgnoresTheFormatsOfADisabledVectorStore(t *testing.T) {
	cfg := &RouterConfig{
		VectorStore: &VectorStoreConfig{
			Enabled:          false,
			SupportedFormats: []string{".pdf"},
		},
	}
	assert.NoError(t, validateConfigStructure(cfg))
}
