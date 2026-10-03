package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const judgeTasksQuery = datasetExportQuery + "&blinding_key=judge-key"

func sha256Hex(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

func openAIChatBody(text string) string {
	return `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"primary-model",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"` + text + `"},"finish_reason":"stop"}]}`
}

// judgedReplayRecord is a compared observation whose stored bodies and shadow
// excerpt carry the text its digests were taken from.
func judgedReplayRecord(t *testing.T, id, responseBody, excerpt string) routerreplay.RoutingRecord {
	t.Helper()
	record := comparedReplayRecord(t, id, "vault")
	record.ResponseBody = responseBody
	record.Outcomes[0].Metadata["response_sha256"] = sha256Hex("primary answer")
	record.Outcomes[1].Metadata["response_sha256"] = sha256Hex("shadow answer")
	record.Outcomes[1].Metadata["response_excerpt"] = excerpt
	return record
}

func judgeTasks(router *OpenAIRouter, query string) *ext_proc.ProcessingResponse {
	return router.handleRouterReplayAPI("GET", "/api/v1/observability/replays/dataset/judge-tasks"+query)
}

func judgeTaskCounts(t *testing.T, response *ext_proc.ProcessingResponse) map[string]interface{} {
	t.Helper()
	assertDatasetStatus(t, response, typev3.StatusCode_OK)
	counts, ok := decodeJSONBody(t, response.GetImmediateResponse().Body)["counts"].(map[string]interface{})
	if !ok {
		t.Fatal("expected judge task counts")
	}
	return counts
}

// The route reads the text behind each digest from the record itself: the
// request body, the assistant text of the stored response, and the shadow
// excerpt. The tasks carry that text and no model name.
func TestRouterReplayJudgeTasksCarryTheRecordedText(t *testing.T) {
	record := judgedReplayRecord(t, "replay-1", openAIChatBody("primary answer"), "shadow answer")
	response := judgeTasks(newDatasetExportRouter(t, record), judgeTasksQuery)

	counts := judgeTaskCounts(t, response)
	assertIntField(t, counts, "pairs", 1)
	assertIntField(t, counts, "tasks", 2)
	body := string(response.GetImmediateResponse().Body)
	for _, want := range []string{"primary answer", "shadow answer", "replay-1"} {
		if !strings.Contains(body, want) {
			t.Fatalf("judge tasks lack %q: %s", want, body)
		}
	}
	for _, identity := range []string{"candidate-model", "judge-key"} {
		if strings.Contains(body, identity) {
			t.Fatalf("judge tasks carry %q: %s", identity, body)
		}
	}
}

// The record does not say which client format its response body is in, so the
// route tries each and keeps the decoding that hashes to the recorded digest.
func TestRouterReplayJudgeTasksReadAnAnthropicResponse(t *testing.T) {
	anthropic := `{"id":"msg_1","type":"message","role":"assistant","model":"primary-model",` +
		`"content":[{"type":"text","text":"primary answer"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":1,"output_tokens":2}}`
	record := judgedReplayRecord(t, "replay-1", anthropic, "shadow answer")

	counts := judgeTaskCounts(t, judgeTasks(newDatasetExportRouter(t, record), judgeTasksQuery))
	assertIntField(t, counts, "pairs", 1)
}

// A truncated excerpt, or a primary a response-stage plugin rewrote after the
// digest was taken, does not hash back and is excluded under a counted reason.
func TestRouterReplayJudgeTasksExcludeTextThatNoLongerMatches(t *testing.T) {
	router := newDatasetExportRouter(t,
		judgedReplayRecord(t, "replay-1", openAIChatBody("primary answer"), "shadow"),
		judgedReplayRecord(t, "replay-2", openAIChatBody("[warning] primary answer"), "shadow answer"),
	)

	counts := judgeTaskCounts(t, judgeTasks(router, judgeTasksQuery))
	assertIntField(t, counts, "pairs", 0)
	excluded, ok := counts["excluded"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected an exclusion tally, got %#v", counts["excluded"])
	}
	assertIntField(t, excluded, "arm_text_digest_mismatch", 2)
}

func TestRouterReplayJudgeTasksCountAnUncapturedShadow(t *testing.T) {
	record := judgedReplayRecord(t, "replay-1", openAIChatBody("primary answer"), "")
	counts := judgeTaskCounts(t, judgeTasks(newDatasetExportRouter(t, record), judgeTasksQuery))

	excluded, ok := counts["excluded"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected an exclusion tally, got %#v", counts["excluded"])
	}
	assertIntField(t, excluded, "arm_text_missing", 1)
}

func TestRouterReplayJudgeTasksRejectAKeyThatBlindsNothing(t *testing.T) {
	router := newDatasetExportRouter(t,
		judgedReplayRecord(t, "replay-1", openAIChatBody("primary answer"), "shadow answer"))

	for name, query := range map[string]string{
		"no key":            datasetExportQuery,
		"the manifest seed": datasetExportQuery + "&blinding_key=seed-a",
	} {
		t.Run(name, func(t *testing.T) {
			assertDatasetStatus(t, judgeTasks(router, query), typev3.StatusCode_BadRequest)
		})
	}
	response := router.handleRouterReplayAPI("POST", "/api/v1/observability/replays/dataset/judge-tasks"+judgeTasksQuery)
	assertDatasetStatus(t, response, typev3.StatusCode_MethodNotAllowed)
}
