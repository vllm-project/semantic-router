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

package selection

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const defaultLLMRouterQueryTemplate = `Decision name: {{.decision_name}}
{{- if .decision_description }}
Decision description: {{.decision_description}}
{{- end }}
{{- if .request_id }}
Request ID: {{.request_id}}
{{- end }}
{{- if .user_id }}
User ID: {{.user_id}}
{{- end }}
{{- if .session_id }}
Session ID: {{.session_id}}
{{- end }}
{{- if .category_name }}
Category: {{.category_name}}
{{- end }}
{{- if .matched_domains }}
Matched domains: {{ join .matched_domains ", " }}
{{- end }}
{{- if .matched_keywords }}
Matched keywords: {{ join .matched_keywords ", " }}
{{- end }}
{{- if .labels }}
Labels: {{ join .labels ", " }}
{{- end }}
{{- if .conversation_history }}
Conversation history:
{{- range .conversation_history }}
- {{ . }}
{{- end }}
{{- end }}
Query:
{{.query}}

Available models:
{{- range .available_models }}
- {{ .name }}{{if .lora_name}} (LoRA: {{ .lora_name }}){{end}}{{if .reasoning_description}} - {{ .reasoning_description }}{{end}}{{if .reasoning_effort}} (effort={{ .reasoning_effort }}){{end}}{{if .weight_note}} {{ .weight_note }}{{end}}
{{- end }}

Think step by step about which model would be best, then route the query.`

// LLMRouterPromptRenderer renders the runtime query sent to the LLM router.
type LLMRouterPromptRenderer struct {
	tmpl *template.Template
}

// NewLLMRouterPromptRenderer builds the query template from either a template
// file or an inline YAML string. When both are empty, it falls back to the
// repository default template.
func NewLLMRouterPromptRenderer(cfg *RLDrivenConfig) (*LLMRouterPromptRenderer, error) {
	source := defaultLLMRouterQueryTemplate

	if cfg != nil {
		if filePath := strings.TrimSpace(cfg.LLMRouterQueryTemplateFile); filePath != "" {
			templateBytes, err := os.ReadFile(filePath)
			if err != nil {
				return nil, fmt.Errorf("failed to read llm router query template file %q: %w", filePath, err)
			}
			source = string(templateBytes)
		} else if inline := strings.TrimSpace(cfg.LLMRouterQueryTemplate); inline != "" {
			source = inline
		}
	}

	tmpl, err := template.New("llm_router_query").Funcs(template.FuncMap{
		"join": strings.Join,
	}).Option("missingkey=zero").Parse(source)
	if err != nil {
		return nil, fmt.Errorf("failed to parse llm router query template: %w", err)
	}

	return &LLMRouterPromptRenderer{
		tmpl: tmpl,
	}, nil
}

// Render executes the prompt template with the provided selection context.
func (r *LLMRouterPromptRenderer) Render(selCtx *SelectionContext) (string, error) {
	if r == nil || r.tmpl == nil {
		return "", fmt.Errorf("llm router query renderer is not configured")
	}

	data := buildLLMRouterTemplateData(selCtx)
	var buf bytes.Buffer
	if err := r.tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute llm router query template: %w", err)
	}

	return strings.TrimSpace(buf.String()), nil
}

func buildLLMRouterTemplateData(selCtx *SelectionContext) map[string]any {
	if selCtx == nil {
		return map[string]any{}
	}

	availableModels := buildLLMRouterAvailableModels(selCtx.CandidateModels)
	availableModelNames := make([]string, 0, len(selCtx.CandidateModels))
	for _, model := range selCtx.CandidateModels {
		if strings.TrimSpace(model.Model) == "" {
			continue
		}
		availableModelNames = append(availableModelNames, model.Model)
	}

	metadata := map[string]any{
		"candidate_count":       len(selCtx.CandidateModels),
		"category_name":         selCtx.CategoryName,
		"cost_weight":           selCtx.CostWeight,
		"quality_weight":        selCtx.QualityWeight,
		"available_model_names": availableModelNames,
	}

	if selCtx.AgenticSession != nil {
		metadata["previous_model"] = selCtx.AgenticSession.PreviousModel
		metadata["previous_response_id"] = selCtx.AgenticSession.PreviousResponseID
		metadata["turn_index"] = selCtx.AgenticSession.TurnIndex
		metadata["history_token_count"] = selCtx.AgenticSession.HistoryTokens
		metadata["cache_warmth"] = selCtx.AgenticSession.CacheWarmth
		metadata["cache_warmth_ok"] = selCtx.AgenticSession.CacheWarmthOK
		metadata["idle_for_seconds"] = selCtx.AgenticSession.IdleFor.Seconds()
		metadata["idle_known"] = selCtx.AgenticSession.IdleKnown
		metadata["phase"] = string(selCtx.AgenticSession.Phase)
		metadata["active_tool_loop"] = selCtx.AgenticSession.ActiveToolLoop
		metadata["has_non_portable_context"] = selCtx.AgenticSession.HasNonPortableContext
	}

	return map[string]any{
		"decision_name":           selCtx.DecisionName,
		"decision_description":    selCtx.DecisionDescription,
		"query":                   selCtx.Query,
		"category_name":           selCtx.CategoryName,
		"conversation_history":    append([]string(nil), selCtx.ConversationHistory...),
		"matched_domains":         append([]string(nil), selCtx.MatchedDomains...),
		"matched_keywords":        append([]string(nil), selCtx.MatchedKeywords...),
		"labels":                  append([]string(nil), selCtx.Labels...),
		"tags":                    append([]string(nil), selCtx.Tags...),
		"user_id":                 selCtx.UserID,
		"session_id":              selCtx.SessionID,
		"request_id":              selCtx.RequestID,
		"previous_model":          previousModel(selCtx),
		"previous_response_id":    previousResponseID(selCtx),
		"turn_index":              turnIndex(selCtx),
		"history_token_count":     historyTokenCount(selCtx),
		"cache_warmth":            cacheWarmth(selCtx),
		"cache_warmth_ok":         cacheWarmthOK(selCtx),
		"session_idle_seconds":    sessionIdleSeconds(selCtx),
		"session_idle_known":      sessionIdleKnown(selCtx),
		"available_models":        availableModels,
		"available_model_names":   availableModelNames,
		"available_model_details": availableModels,
		"metadata":                metadata,
	}
}

func buildLLMRouterAvailableModels(models []config.ModelRef) []map[string]any {
	available := make([]map[string]any, 0, len(models))
	for idx, model := range models {
		available = append(available, map[string]any{
			"index":                 idx,
			"name":                  model.Model,
			"model":                 model.Model,
			"lora_name":             model.LoRAName,
			"weight":                model.Weight,
			"reasoning_description": model.ReasoningDescription,
			"reasoning_effort":      model.ReasoningEffort,
			"use_reasoning":         model.UseReasoning != nil && *model.UseReasoning,
			"weight_note":           modelWeightNote(model.Weight),
		})
	}
	return available
}

func modelWeightNote(weight float64) string {
	if weight == 0 {
		return ""
	}
	return fmt.Sprintf("(weight=%.3f)", weight)
}

func previousModel(selCtx *SelectionContext) string {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return ""
	}
	return selCtx.AgenticSession.PreviousModel
}

func previousResponseID(selCtx *SelectionContext) string {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return ""
	}
	return selCtx.AgenticSession.PreviousResponseID
}

func turnIndex(selCtx *SelectionContext) int {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return 0
	}
	return selCtx.AgenticSession.TurnIndex
}

func historyTokenCount(selCtx *SelectionContext) int {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return 0
	}
	return selCtx.AgenticSession.HistoryTokens
}

func cacheWarmth(selCtx *SelectionContext) float64 {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return 0
	}
	return selCtx.AgenticSession.CacheWarmth
}

func cacheWarmthOK(selCtx *SelectionContext) bool {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return false
	}
	return selCtx.AgenticSession.CacheWarmthOK
}

func sessionIdleSeconds(selCtx *SelectionContext) float64 {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return 0
	}
	return selCtx.AgenticSession.IdleFor.Seconds()
}

func sessionIdleKnown(selCtx *SelectionContext) bool {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return false
	}
	return selCtx.AgenticSession.IdleKnown
}
