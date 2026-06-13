package classification

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiStepPipelineKeywordFragmentDetectsStepMarkers(t *testing.T) {
	rules := loadMultiStepPipelineKeywordRules(t)
	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier() error = %v", err)
	}
	defer classifier.Free()

	tests := []struct {
		name        string
		text        string
		wantRule    string
		wantKeyword string
	}{
		{
			name:        "summarize marker",
			text:        "<semantic-router-step>__pipeline_step:summarize__</semantic-router-step>",
			wantRule:    "pipeline_step_summarize",
			wantKeyword: "__pipeline_step:summarize__",
		},
		{
			name:        "summarize alias",
			text:        "<semantic-router-step>__step1__</semantic-router-step>",
			wantRule:    "pipeline_step_summarize",
			wantKeyword: "__step1__",
		},
		{
			name:        "summarize marker is case insensitive",
			text:        "<semantic-router-step>__PIPELINE_STEP:SUMMARIZE__</semantic-router-step>",
			wantRule:    "pipeline_step_summarize",
			wantKeyword: "__pipeline_step:summarize__",
		},
		{
			name:        "extract marker",
			text:        "<semantic-router-step>__pipeline_step:extract__</semantic-router-step>",
			wantRule:    "pipeline_step_extract",
			wantKeyword: "__pipeline_step:extract__",
		},
		{
			name:        "extract alias",
			text:        "<semantic-router-step>__step2__</semantic-router-step>",
			wantRule:    "pipeline_step_extract",
			wantKeyword: "__step2__",
		},
		{
			name:        "extract marker is case insensitive",
			text:        "<semantic-router-step>__PIPELINE_STEP:EXTRACT__</semantic-router-step>",
			wantRule:    "pipeline_step_extract",
			wantKeyword: "__pipeline_step:extract__",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotRule, gotKeywords, err := classifier.ClassifyWithKeywords(tt.text)
			if err != nil {
				t.Fatalf("ClassifyWithKeywords() error = %v", err)
			}
			if gotRule != tt.wantRule {
				t.Fatalf("ClassifyWithKeywords() rule = %q, want %q", gotRule, tt.wantRule)
			}
			if !containsString(gotKeywords, tt.wantKeyword) {
				t.Fatalf("ClassifyWithKeywords() keywords = %v, want to include %q", gotKeywords, tt.wantKeyword)
			}
		})
	}
}

func loadMultiStepPipelineKeywordRules(t *testing.T) []config.KeywordRule {
	t.Helper()
	path := filepath.Join(repoRootFromClassificationTest(t), "config", "signal", "keyword", "multi-step-pipeline.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read multi-step pipeline keyword fragment: %v", err)
	}

	var fragment struct {
		Routing struct {
			Signals struct {
				Keywords []config.KeywordRule `yaml:"keywords"`
			} `yaml:"signals"`
		} `yaml:"routing"`
	}
	if err := yaml.Unmarshal(data, &fragment); err != nil {
		t.Fatalf("parse multi-step pipeline keyword fragment: %v", err)
	}
	if len(fragment.Routing.Signals.Keywords) != 2 {
		t.Fatalf("keyword fragment rules = %d, want 2", len(fragment.Routing.Signals.Keywords))
	}
	return fragment.Routing.Signals.Keywords
}

func repoRootFromClassificationTest(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", ".."))
}
