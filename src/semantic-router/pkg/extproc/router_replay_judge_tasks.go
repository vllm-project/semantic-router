package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/shadowdataset"
)

const routerReplayJudgeTasksPath = routerReplayDatasetPath + "/judge-tasks"

// storedResponseFormats are the client formats a stored response body can be
// in. The record does not keep the one the client used.
var storedResponseFormats = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1,
	llmprotocol.OpenAIResponsesV1,
	llmprotocol.AnthropicMessagesV1,
}

// handleRouterReplayJudgeTasksAPI serves
// GET /api/v1/observability/replays/dataset/judge-tasks. It builds the same
// manifest the dataset export does and hands out blinded pairwise tasks over
// it. The tasks carry captured text, so the route needs replay.detail.
func (r *OpenAIRouter) handleRouterReplayJudgeTasksAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	values, manifest, records, failure := r.buildRouterReplayDataset(method, rawQuery)
	if failure != nil {
		return failure
	}
	tasks, err := shadowdataset.BuildJudgeTasks(manifest, r.judgeTaskTexts(manifest, records),
		strings.TrimSpace(values.Get("blinding_key")))
	if err != nil {
		return r.createErrorResponse(400, err.Error())
	}
	return r.createRouterReplayJSONResponse(200, tasks)
}

// judgeTaskTexts reads the text behind each example's digests. The input is the
// stored request body, the exact bytes the input digest covers. The primary is
// the assistant text of the stored response, and each shadow is the excerpt the
// shadow dispatch plugin captured. BuildJudgeTasks decides whether each text is
// whole by hashing it, so nothing here needs to.
func (r *OpenAIRouter) judgeTaskTexts(
	manifest shadowdataset.Manifest,
	records []routerreplay.RoutingRecord,
) map[string]shadowdataset.ExampleText {
	byReplay := make(map[string]routerreplay.RoutingRecord, len(records))
	for _, record := range records {
		byReplay[record.ID] = record
	}
	texts := make(map[string]shadowdataset.ExampleText, len(manifest.Examples))
	for _, example := range manifest.Examples {
		record, found := byReplay[example.Lineage.ReplayID]
		if !found {
			continue
		}
		text := shadowdataset.ExampleText{
			Input:   record.RequestBody,
			Primary: r.storedAnswerText(record.ResponseBody),
		}
		for _, arm := range example.Shadows {
			text.Shadows = append(text.Shadows, shadowExcerpt(record, arm))
		}
		texts[example.ID] = text
	}
	return texts
}

// storedAnswerText decodes a stored response body into assistant text, trying
// each client format in turn. A decoding that reads the wrong text is caught
// downstream, because BuildJudgeTasks hashes it against the recorded digest.
func (r *OpenAIRouter) storedAnswerText(body string) string {
	engine, err := r.protocolEngine()
	if err != nil || body == "" {
		return ""
	}
	for _, format := range storedResponseFormats {
		decoded, decodeErr := engine.TranslateResponse(format, format, []byte(body), nil)
		if decodeErr != nil {
			continue
		}
		if text := semanticResponseText(decoded.Response); text != "" {
			return text
		}
	}
	return ""
}

// shadowExcerpt finds the captured text of one shadow arm. Arms are matched by
// the digest the plugin recorded, because one candidate may answer twice and
// the model name alone would not say which answer an arm stands for.
func shadowExcerpt(record routerreplay.RoutingRecord, arm shadowdataset.Arm) string {
	for _, outcome := range record.Outcomes {
		if outcome.Source == shadowDispatchOutcomeSource &&
			outcome.Verdict == shadowVerdictCompleted &&
			outcome.Metadata["response_sha256"] == arm.OutputDigest {
			return outcome.Metadata["response_excerpt"]
		}
	}
	return ""
}
