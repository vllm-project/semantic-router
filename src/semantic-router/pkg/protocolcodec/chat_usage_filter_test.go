package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatUsageStreamFilterPreservesExtensionsAndArbitraryChunking(t *testing.T) {
	input := []byte(
		"data: {\"id\":\"chunk_1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3},\"provider_extension\":{\"trace\":\"keep\"}}\n\n" +
			"data: {\"id\":\"chunk_1\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n" +
			"data: [DONE]\n\n",
	)
	filter := NewChatUsageStreamFilter(1 << 20)
	var output []byte
	for offset := 0; offset < len(input); {
		end := offset + 1 + offset%13
		if end > len(input) {
			end = len(input)
		}
		chunk, err := filter.Push(input[offset:end])
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, chunk...)
		offset = end
	}
	if bytes.Contains(output, []byte("data: [DONE]")) {
		t.Fatalf("success terminal escaped before clean finalization: %s", output)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	output = append(output, final...)
	if bytes.Contains(output, []byte(`"usage"`)) {
		t.Fatalf("public stream retained internal usage evidence: %s", output)
	}
	if !bytes.Contains(output, []byte(`"provider_extension":{"trace":"keep"}`)) {
		t.Fatalf("public stream lost provider extension: %s", output)
	}
	if count := bytes.Count(output, []byte("data: [DONE]")); count != 1 {
		t.Fatalf("terminal frame count = %d, want 1: %s", count, output)
	}
}

func TestChatUsageStreamFilterRejectsMalformedUsageFrame(t *testing.T) {
	filter := NewChatUsageStreamFilter(1 << 20)
	if _, err := filter.Push([]byte("data: {\"choices\":{},\"usage\":{}}\n\n")); err == nil {
		t.Fatal("malformed choices object was accepted")
	}
}

func TestChatUsageStreamFilterRejectsLateBOMAndEmptyData(t *testing.T) {
	valid := []byte("data: {\"id\":\"response_1\",\"choices\":[]}\n\n")
	filter := NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if _, err := filter.Push(valid); err != nil {
		t.Fatal(err)
	}
	lateBOM := append([]byte{0xef, 0xbb, 0xbf}, valid...)
	if _, err := filter.Push(lateBOM); err == nil {
		t.Fatal("late UTF-8 BOM was accepted")
	}

	filter = NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if _, err := filter.Push([]byte("data:\n\n")); err == nil {
		t.Fatal("empty data event was treated as a keepalive")
	}
}

func TestChatUsageStreamFilterRejectsDataAndTrailingFragmentsAfterTerminal(t *testing.T) {
	terminal := []byte("data: [DONE]\n\n")
	data := []byte("data: {\"id\":\"response_1\",\"choices\":[]}\n\n")

	filter := NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if output, err := filter.Push(terminal); err != nil || len(output) != 0 {
		t.Fatalf("terminal was not held: output=%q err=%v", output, err)
	}
	_, firstErr := filter.Push(data)
	assertProtocolError(t, firstErr, llmprotocol.ErrorUpstreamUnavailable, "stream_event_after_terminal")
	_, repeatedErr := filter.Push(data)
	if goldenProtocolErrorFrom(t, firstErr) != goldenProtocolErrorFrom(t, repeatedErr) {
		t.Fatalf("poisoned filter changed failure: first=%v repeated=%v", firstErr, repeatedErr)
	}

	filter = NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if output, err := filter.Push(terminal); err != nil || len(output) != 0 {
		t.Fatalf("terminal was not held: output=%q err=%v", output, err)
	}
	if output, err := filter.Push([]byte("data: {")); err != nil || len(output) != 0 {
		t.Fatalf("partial trailing frame failed before EOF: output=%q err=%v", output, err)
	}
	if output, err := filter.Finalize(); err == nil || len(output) != 0 {
		t.Fatalf("malformed trailing frame did not suppress success: output=%q err=%v", output, err)
	}
}
