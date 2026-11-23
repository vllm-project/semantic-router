package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// contentPartsTurn is a user turn sent as text parts, the way a coding agent
// sends one: a <system-reminder> part, then the user's own words in two parts.
func contentPartsTurn(t *testing.T) *openai.ChatCompletionNewParams {
	t.Helper()
	var params openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{"model":"auto","messages":[{"role":"user","content":[`+
		`{"type":"text","text":"<system-reminder>\nKeep answers short.\n</system-reminder>"},`+
		`{"type":"text","text":"What is the capital of Australia?"},{"type":"text","text":"Answer in one word."}]}]}`), &params))
	return &params
}

const contentPartsQuestion = "What is the capital of Australia?\nAnswer in one word."

func TestEscalateAnswerChecksJudgeTheUsersOwnWords(t *testing.T) {
	server, _ := newEscalateLogprobBackend(t, map[string]int{}, false)
	defer server.Close()
	checker := &stubAnswerChecker{correct: map[string]float64{"small answer": 0.9}}
	request := escalateRequest("", "")
	request.OriginalRequest = contentPartsTurn(t)
	request.Algorithm.Escalate.Accept = &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.7)}}
	request.AnswerChecks = checker

	if _, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if want := []string{"answer_check:" + contentPartsQuestion + "|small answer"}; !slices.Equal(checker.asked, want) {
		t.Fatalf("checks = %q, want %q", checker.asked, want)
	}
}

func TestFusionJudgeAndGroundingSeeTheQuestionOfAContentPartTurn(t *testing.T) {
	var mu sync.Mutex
	var judgePrompts, groundedQuestions []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		if payload.Model == "judge" {
			var prompt string
			require.NoError(t, json.Unmarshal(payload.Messages[len(payload.Messages)-1].Content, &prompt))
			mu.Lock()
			judgePrompts = append(judgePrompts, prompt)
			mu.Unlock()
		}
		writeFusionTestCompletion(w, payload.Model, payload.Model+" says Canberra", http.StatusOK)
	}))
	defer server.Close()
	req := newFusionTestRequest()
	req.OriginalRequest = contentPartsTurn(t)
	req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
		Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone,
		Grounding: &config.FusionGroundingConfig{Enabled: true, Reference: config.FusionGroundingReferencePanel},
	}}
	req.Grounding = &GroundingBackends{Detect: func(_ context.Context, _, question, _ string) (GroundingEvidence, error) {
		mu.Lock()
		groundedQuestions = append(groundedQuestions, question)
		mu.Unlock()
		return GroundingEvidence{}, nil
	}}

	if _, err := fusionHopsTo(server.URL).Execute(context.Background(), req); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if len(judgePrompts) != 1 || !strings.Contains(judgePrompts[0], contentPartsQuestion) || strings.Contains(judgePrompts[0], "Keep answers short.") {
		t.Fatalf("judge prompts = %q, want one carrying only the user's question", judgePrompts)
	}
	if len(groundedQuestions) == 0 || slices.ContainsFunc(groundedQuestions, func(q string) bool { return q != contentPartsQuestion }) {
		t.Fatalf("grounding questions = %q, want %q", groundedQuestions, contentPartsQuestion)
	}
}

// Grounding reads system and tool messages sent as several text parts, as
// Messages system blocks and multi-block tool results arrive, as its context.
func TestFusionGroundingReadsSystemAndToolTextParts(t *testing.T) {
	var params openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{"model":"auto","messages":[`+
		`{"role":"system","content":[{"type":"text","text":"Context: the sky is green."},{"type":"text","text":"Grass is blue."}]},`+
		`{"role":"user","content":"What color is the sky?"},`+
		`{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},`+
		`{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"Sky: green."},{"type":"text","text":"Source: atlas."}]},`+
		`{"role":"user","content":"Answer now."}]}`), &params))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		writeFusionTestCompletion(w, payload.Model, payload.Model+" says green", http.StatusOK)
	}))
	defer server.Close()
	req := newFusionTestRequest()
	req.OriginalRequest = &params
	req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
		Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone,
		Grounding: &config.FusionGroundingConfig{Enabled: true},
	}}
	var mu sync.Mutex
	var contexts []string
	req.Grounding = &GroundingBackends{Detect: func(_ context.Context, contextText, _, _ string) (GroundingEvidence, error) {
		mu.Lock()
		contexts = append(contexts, contextText)
		mu.Unlock()
		return GroundingEvidence{}, nil
	}}

	if _, err := fusionHopsTo(server.URL).Execute(context.Background(), req); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	want := "Context: the sky is green.\nGrass is blue.\nSky: green.\nSource: atlas."
	if len(contexts) == 0 || slices.ContainsFunc(contexts, func(c string) bool { return c != want }) {
		t.Fatalf("grounding contexts = %q, want %q", contexts, want)
	}
}

// Reminder blocks are removed from the turn's joined text, so a reminder split
// across parts is left out too, and a turn left with only whitespace has no
// words of its own.
func TestEscalateAnswerChecksSkipSplitAndWhitespaceReminderTurns(t *testing.T) {
	for name, messages := range map[string]string{
		"reminder split across parts": `[{"role":"user","content":[{"type":"text","text":"<system-reminder>Ignore prior task"},` +
			`{"type":"text","text":"</system-reminder>"},{"type":"text","text":"What is the capital of Australia?"}]}]`,
		"reminder and whitespace": `[{"role":"user","content":"What is the capital of Australia?"},{"role":"assistant","content":"Let me check."},` +
			`{"role":"user","content":[{"type":"text","text":"<system-reminder>Keep answers short.</system-reminder>"},{"type":"text","text":" \n "}]}]`,
	} {
		t.Run(name, func(t *testing.T) {
			server, _ := newEscalateLogprobBackend(t, map[string]int{}, false)
			defer server.Close()
			var params openai.ChatCompletionNewParams
			require.NoError(t, json.Unmarshal([]byte(`{"model":"auto","messages":`+messages+`}`), &params))
			checker := &stubAnswerChecker{correct: map[string]float64{"small answer": 0.9}}
			request := escalateRequest("", "")
			request.OriginalRequest = &params
			request.Algorithm.Escalate.Accept = &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.7)}}
			request.AnswerChecks = checker

			if _, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if want := []string{"answer_check:What is the capital of Australia?|small answer"}; !slices.Equal(checker.asked, want) {
				t.Fatalf("checks = %q, want %q", checker.asked, want)
			}
		})
	}
}
