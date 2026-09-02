package testcases

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("dynamo-nvext-chat-contract", pkgtestcases.TestCase{
		Description: "Dynamo nvext survives Chat Completions request, buffered response, and SSE response paths",
		Tags:        []string{"dynamo", "nvext", "chat-completions", "streaming"},
		Fn:          testDynamoNVExtChatContract,
	})
}

func testDynamoNVExtChatContract(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	buffered, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", dynamoNVExtRequest(false), false)
	if err != nil {
		return fmt.Errorf("buffered Dynamo nvext request: %w", err)
	}
	if err := verifyBufferedDynamoNVExt(buffered); err != nil {
		return err
	}

	streamed, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", dynamoNVExtRequest(true), true)
	if err != nil {
		return fmt.Errorf("streaming Dynamo nvext request: %w", err)
	}
	if err := verifyStreamedDynamoNVExt(streamed); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"buffered": true, "streaming": true, "backend_type": "dynamo"})
	}
	return nil
}

func dynamoNVExtRequest(stream bool) map[string]any {
	return map[string]any{
		"model": "openai/gpt-oss-20b", "stream": stream,
		"messages":   []map[string]string{{"role": "user", "content": "verify Dynamo nvext"}},
		"cache_salt": "top-level-cache",
		"nvext": map[string]any{
			"cache_salt":   "nested-cache",
			"extra_fields": []string{"worker_id", "timing", "completion_token_ids"},
		},
	}
}

func verifyBufferedDynamoNVExt(body []byte) error {
	var response struct {
		NVExt json.RawMessage `json:"nvext"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode buffered Chat response: %w", err)
	}
	return verifyDynamoNVExtPayload(response.NVExt)
}

func verifyStreamedDynamoNVExt(body []byte) error {
	found := 0
	for _, data := range protocolSSEDataFrames(body) {
		if data == "[DONE]" {
			continue
		}
		var chunk struct {
			NVExt json.RawMessage `json:"nvext"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode Chat SSE chunk: %w", err)
		}
		if len(chunk.NVExt) == 0 {
			continue
		}
		found++
		if err := verifyDynamoNVExtPayload(chunk.NVExt); err != nil {
			return err
		}
	}
	if found != 1 {
		return fmt.Errorf("Dynamo Chat stream contained %d nvext chunks, want 1: %s", found, truncateString(string(body), 1200))
	}
	return nil
}

func verifyDynamoNVExtPayload(raw json.RawMessage) error {
	var extension struct {
		WorkerID struct {
			PrefillWorkerID uint64 `json:"prefill_worker_id"`
			DecodeWorkerID  uint64 `json:"decode_worker_id"`
		} `json:"worker_id"`
		Timing struct {
			RequestReceivedMS uint64  `json:"request_received_ms"`
			TTFTMS            float64 `json:"ttft_ms"`
		} `json:"timing"`
		CompletionTokenIDs []uint32 `json:"completion_token_ids"`
	}
	if err := json.Unmarshal(raw, &extension); err != nil {
		return fmt.Errorf("decode Dynamo nvext: %w", err)
	}
	if extension.WorkerID.PrefillWorkerID != 11 || extension.WorkerID.DecodeWorkerID != 22 ||
		extension.Timing.RequestReceivedMS != 1000 || extension.Timing.TTFTMS != 3.75 ||
		len(extension.CompletionTokenIDs) != 2 || extension.CompletionTokenIDs[0] != 101 || extension.CompletionTokenIDs[1] != 102 {
		return fmt.Errorf("unexpected Dynamo nvext payload: %s", raw)
	}
	return nil
}
