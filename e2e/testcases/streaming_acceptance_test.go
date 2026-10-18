package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func TestStreamingCachePrimeUsesCompleteStreamingResponse(t *testing.T) {
	const replay = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	for _, tc := range []struct {
		name, contentType, body string
		wantError               bool
	}{
		{"complete stream", "text/event-stream", replay, false},
		{"buffered response", "application/json", `{"choices":[{"message":{"content":"answer"}}]}`, true},
		{"wrong content type", "application/json", replay, true},
		{"empty stream", "text/event-stream", "data: [DONE]\n\n", true},
		{"missing terminal", "text/event-stream", strings.ReplaceAll(replay, "data: [DONE]\n\n", ""), true},
		{"missing finish", "text/event-stream", strings.ReplaceAll(replay, "\"stop\"", "null"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Model  string `json:"model"`
					Stream bool   `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode priming request: %v", err)
				}
				if r.URL.Path != "/v1/chat/completions" || !request.Stream || request.Model != "vllm-sr/auto" {
					t.Errorf("priming request must preserve streaming cache identity: path=%s request=%+v", r.URL.Path, request)
				}
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			endpoint, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			_, text, err := primeStreamingResponseCache(context.Background(), "question", "vllm-sr/auto", endpoint.Port())
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
			if !tc.wantError && text != "answer" {
				t.Fatalf("priming output = %q, want answer", text)
			}
		})
	}
}

func TestStreamingSafetyRejectsUnrelatedFastResponses(t *testing.T) {
	for _, tc := range []struct {
		name, decision, fast, body, wantDecision, refusal string
		wantError                                         bool
	}{
		{"refusal", "block_pii", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", false},
		{"other blocker", "block_jailbreak", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"wrong refusal", "block_pii", "true", `{"choices":[{"message":{"content":"unrelated"}}]}`, "block_pii", "denied", true},
		{"no decision", "", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"invalid body", "block_pii", "true", `not json`, "block_pii", "denied", true},
		{"no fast response", "block_pii", "", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"benign control", "general_decision", "", `{"choices":[{"message":{"content":"answer"}}]}`, "general_decision", "", false},
		{"universal blocking", "general_decision", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "general_decision", "", true},
		{"empty control", "general_decision", "", `{"choices":[{"message":{"content":""}}]}`, "general_decision", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := make(http.Header)
			headers.Set("x-vsr-selected-decision", tc.decision)
			headers.Set("x-vsr-fast-response", tc.fast)
			err := validateStreamingSafetyResponse(headers, []byte(tc.body), tc.wantDecision, tc.refusal)
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
		})
	}
}

func TestStreamingCacheRequiresActualSSEReplay(t *testing.T) {
	const replay = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	const done = "data: [DONE]\n\n"
	const finish = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	const content = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"
	const usage = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n"
	for _, tc := range []struct {
		name, hit, contentType, body string
		wantError                    bool
	}{
		{"complete replay", "true", "text/event-stream; charset=utf-8", replay, false},
		{"usage after finish", "true", "text/event-stream", strings.Replace(replay, done, usage+done, 1), false},
		{"content after finish", "true", "text/event-stream", finish + content + done, true},
		{"duplicate finish", "true", "text/event-stream", strings.Replace(replay, done, finish+done, 1), true},
		{"error finish overwritten", "true", "text/event-stream", strings.Replace(finish, "stop", "error", 1) + replay, true},
		{"empty choice after finish", "true", "text/event-stream", strings.Replace(replay, done, strings.Replace(finish, "\"stop\"", "null", 1)+done, 1), true},
		{"not cached", "false", "text/event-stream", replay, true},
		{"JSON only", "true", "application/json", `{"choices":[]}`, true},
		{"empty stream", "true", "text/event-stream", "", true},
		{"truncated stream", "true", "text/event-stream", strings.ReplaceAll(replay, "data: [DONE]\n\n", ""), true},
		{"wrong content", "true", "text/event-stream", strings.ReplaceAll(replay, "answer", "other"), true},
		{"extra content", "true", "text/event-stream", strings.ReplaceAll(replay, "answer", "answer extra"), true},
		{"duplicate terminal", "true", "text/event-stream", replay + "data: [DONE]\n\n", true},
		{"early terminal", "true", "text/event-stream", "data: [DONE]\n\n" + strings.ReplaceAll(replay, "data: [DONE]\n\n", ""), true},
		{"error terminal", "true", "text/event-stream", strings.ReplaceAll(replay, "stop", "error"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := make(http.Header)
			headers.Set("x-vsr-cache-hit", tc.hit)
			headers.Set("Content-Type", tc.contentType)
			err := validateStreamingCacheReplay(headers, []byte(tc.body), "answer")
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
		})
	}
}

func TestStreamingKeywordRoutingRequiresEveryProbe(t *testing.T) {
	for _, tc := range []struct {
		name       string
		decisions  []string
		urgentCode int
		wantError  bool
	}{
		{"all routes and fallback", []string{"code_keywords_decision", "urgent_request_decision", "general_decision"}, http.StatusOK, false},
		{"code route fails", []string{"general_decision", "urgent_request_decision", "general_decision"}, http.StatusOK, true},
		{"urgency route fails", []string{"code_keywords_decision", "general_decision", "general_decision"}, http.StatusOK, true},
		{"universal keyword route", []string{"code_keywords_decision", "code_keywords_decision", "code_keywords_decision"}, http.StatusOK, true},
		{"fallback fails", []string{"code_keywords_decision", "urgent_request_decision", "urgent_request_decision"}, http.StatusOK, true},
		{"missing route header", []string{"code_keywords_decision", "", "general_decision"}, http.StatusOK, true},
		{"one HTTP error", []string{"code_keywords_decision", "urgent_request_decision", "general_decision"}, http.StatusBadRequest, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Messages []struct {
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil || len(request.Messages) != 1 {
					t.Errorf("unexpected probe body: %+v, error %v", request, err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				var index int
				switch request.Messages[0].Content {
				case "Can you help me implement a function to debug this algorithm?":
					index = 0
				case "This is an urgent emergency, I need help immediately!":
					index = 1
				case "Explain why the sky is blue.":
					index = 2
				default:
					t.Errorf("unexpected probe: %q", request.Messages[0].Content)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				requests.Add(1)
				w.Header().Set("x-vsr-selected-decision", tc.decisions[index])
				if index == 1 {
					w.WriteHeader(tc.urgentCode)
				}
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"answer"}}]}`))
			}))
			defer server.Close()
			endpoint, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			err = checkStreamingKeywordRouting(context.Background(), endpoint.Port(), pkgtestcases.TestCaseOptions{})
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
			if count := int(requests.Load()); count != len(tc.decisions) {
				t.Fatalf("ran %d probes, want %d", count, len(tc.decisions))
			}
		})
	}
}

func TestStreamingLargeBodyRequiresExactProviderMessages(t *testing.T) {
	longContext := strings.Repeat("This is padding context to make the body large enough for multi-chunk delivery. ", 1000)
	expected := []map[string]string{
		{"role": "system", "content": longContext},
		{"role": "user", "content": "This is an urgent emergency, I need help immediately!"},
	}
	for _, tc := range []struct {
		name      string
		status    int
		decision  string
		mutate    func(map[string]interface{})
		wantError bool
	}{
		{"complete original messages", http.StatusOK, "urgent_request_decision", nil, false},
		{"arbitrary 400", http.StatusBadRequest, "urgent_request_decision", nil, true},
		{"wrong decision", http.StatusOK, "code_keywords_decision", nil, true},
		{"missing observation", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) { delete(body, "messages") }, true},
		{"missing final message", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) {
			body["messages"] = body["messages"].([]interface{})[:1]
		}, true},
		{"truncated context", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) {
			body["messages"].([]interface{})[0].(map[string]interface{})["content"] = longContext[:65536]
		}, true},
		{"rewritten user message", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) {
			body["messages"].([]interface{})[1].(map[string]interface{})["content"] = "different prompt"
		}, true},
		{"wrong role", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) {
			body["messages"].([]interface{})[0].(map[string]interface{})["role"] = "user"
		}, true},
		{"reordered messages", http.StatusOK, "urgent_request_decision", func(body map[string]interface{}) {
			messages := body["messages"].([]interface{})
			messages[0], messages[1] = messages[1], messages[0]
		}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := map[string]interface{}{"messages": []interface{}{
				map[string]interface{}{"role": "system", "content": longContext},
				map[string]interface{}{"role": "user", "content": expected[1]["content"]},
			}}
			if tc.mutate != nil {
				tc.mutate(body)
			}
			err := validateStreamingLargeBodyDispatch(tc.status, tc.decision, body, expected)
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
		})
	}
}
