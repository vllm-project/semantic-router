package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chunkUsages names what each chunk of a Chat stream carries as its usage:
// "absent", "null" or "totals".
func chunkUsages(t *testing.T, stream []byte) []string {
	t.Helper()
	var usages []string
	for _, data := range sseData(t, stream) {
		var chunk map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &chunk), "%s", data)
		usage, present := chunk["usage"]
		switch {
		case !present:
			usages = append(usages, "absent")
		case string(usage) == "null":
			usages = append(usages, "null")
		default:
			require.JSONEq(t, `[]`, string(chunk["choices"]), "the usage chunk: %s", data)
			usages = append(usages, "totals")
		}
	}
	return usages
}

func requireNullUsageUntilTheUsageChunk(t *testing.T, stream []byte) {
	t.Helper()
	usages := chunkUsages(t, stream)
	require.Greater(t, len(usages), 1, "%s", stream)
	for index, usage := range usages[:len(usages)-1] {
		require.Equal(t, "null", usage, "chunk %d of %s", index, stream)
	}
	require.Equal(t, "totals", usages[len(usages)-1], "%s", stream)
}

func translateToChatClient(t *testing.T, source llmprotocol.WireFormat, body []byte, includeUsage *bool) []byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
		Options: llmprotocol.StreamOptions{IncludeUsage: includeUsage},
	})
	require.NoError(t, err)
	frames, _, _, err := stream.Push(body)
	require.NoError(t, err)
	final, _, _, err := stream.Finalize(nil)
	require.NoError(t, err)
	return bytes.Join(append(frames, final...), nil)
}

// A Chat client that asks for usage gets "usage": null on every chunk before
// the usage chunk, as OpenAI sends it, whatever the backend's format; one that
// does not ask gets no usage member at all.
func TestChatStreamChunksCarryNullUsageWhenTheClientAsksForUsage(t *testing.T) {
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		for name, fixture := range map[string][]byte{"text": streamFixture(source), "tool call": toolStreamFixture(source)} {
			t.Run(string(source)+"/"+name, func(t *testing.T) {
				requireNullUsageUntilTheUsageChunk(t, translateToChatClient(t, source, fixture, llmprotocol.Bool(true)))
				for _, includeUsage := range []*bool{nil, llmprotocol.Bool(false)} {
					for index, usage := range chunkUsages(t, translateToChatClient(t, source, fixture, includeUsage)) {
						require.Equal(t, "absent", usage, "chunk %d", index)
					}
				}
			})
		}
	}
}

// withNullUsage is a chunk frame as a client that asks for usage receives it.
func withNullUsage(frame []byte) []byte {
	return []byte(strings.TrimSuffix(string(frame), "}\n\n") + `,"usage":null}` + "\n\n")
}

// A Chat backend's chunks reach a client that asks for usage with only
// "usage": null added, however the stream is split into reads; a chunk that
// already carries usage, null or totals, passes byte for byte.
func TestChatPublicStreamFilterAddsNullUsageAndKeepsTheBackendsBytes(t *testing.T) {
	first := "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"a < b\"},\"finish_reason\":null}]}\n\n"
	finish := "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n"
	usage := "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n"
	done := "data: [DONE]\n\n"
	input := []byte(first + finish + usage + done)
	want := string(withNullUsage([]byte(first))) + finish + usage + done
	for _, size := range []int{len(input), 7, 1} {
		filter := NewChatPublicStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes, true)
		var output []byte
		for offset := 0; offset < len(input); offset += size {
			chunk, err := filter.Push(input[offset:min(offset+size, len(input))])
			require.NoError(t, err)
			output = append(output, chunk...)
		}
		final, err := filter.Finalize()
		require.NoError(t, err)
		require.Equal(t, want, string(append(output, final...)), "reads of %d bytes", size)
	}
}

// A chunk the filter rewrites for another reason gets the null usage too, and
// a client that does not ask for usage gets none.
func TestChatPublicStreamFilterNullUsageOnRewrittenChunks(t *testing.T) {
	stream := []byte("data: {\"id\":\"c\",\"model\":\"m\",\"system_fingerprint\":\"fp\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n" +
		"data: [DONE]\n\n")
	public := publicChatStream(t, stream)
	requireNullUsageUntilTheUsageChunk(t, public)
	require.NotContains(t, string(public), "system_fingerprint")

	filter := NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	output, err := filter.Push(stream)
	require.NoError(t, err)
	final, err := filter.Finalize()
	require.NoError(t, err)
	for index, usage := range chunkUsages(t, append(output, final...)) {
		require.Equal(t, "absent", usage, "chunk %d", index)
	}
}
