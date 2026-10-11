package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCollectPIIRuleContentsForSource(t *testing.T) {
	tests := []struct {
		name            string
		rule            config.PIIRule
		piiText         string
		nonUserMessages []string
		toolResultTexts []string
		want            []string
	}{
		{
			name:            "legacy source uses current text only",
			rule:            config.PIIRule{},
			piiText:         "current",
			nonUserMessages: []string{"history"},
			toolResultTexts: []string{"tool"},
			want:            []string{"current"},
		},
		{
			name:            "legacy source can include history",
			rule:            config.PIIRule{IncludeHistory: true},
			piiText:         "current",
			nonUserMessages: []string{"history-1", "", "history-2"},
			toolResultTexts: []string{"tool"},
			want:            []string{"current", "history-1", "history-2"},
		},
		{
			name:            "tool result source is isolated",
			rule:            config.PIIRule{Source: config.PIISourceToolResult, IncludeHistory: true},
			piiText:         "current",
			nonUserMessages: []string{"history"},
			toolResultTexts: []string{"tool-1", "", "tool-2", "tool-1"},
			want:            []string{"tool-1", "tool-2", "tool-1"},
		},
		{
			name:            "empty inputs produce no content",
			rule:            config.PIIRule{Source: config.PIISourceToolResult},
			toolResultTexts: []string{"", ""},
			want:            nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := collectPIIRuleContentsForSource(tt.rule, tt.piiText, tt.nonUserMessages, tt.toolResultTexts)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("collectPIIRuleContentsForSource() = %#v, want %#v", got, tt.want)
			}
		})
	}
}
