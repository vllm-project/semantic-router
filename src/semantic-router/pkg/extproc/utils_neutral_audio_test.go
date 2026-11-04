package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestExtractSemanticRequestSignalsInlineUserAudio(t *testing.T) {
	for _, tc := range []struct {
		name    string
		role    llmprotocol.Role
		content llmprotocol.Content
		want    string
	}{
		{"neutral inline", llmprotocol.RoleUser, llmprotocol.Content{Kind: llmprotocol.ContentAudio, MediaType: "audio/wav", Data: "AAAA"}, "data:audio/wav;base64,AAAA"},
		{"data uri", llmprotocol.RoleUser, llmprotocol.Content{Kind: llmprotocol.ContentAudio, URL: "data:audio/wav;base64,AAAA"}, "data:audio/wav;base64,AAAA"},
		{"remote", llmprotocol.RoleUser, llmprotocol.Content{Kind: llmprotocol.ContentAudio, URL: "https://example.com/audio.wav"}, ""},
		{"local", llmprotocol.RoleUser, llmprotocol.Content{Kind: llmprotocol.ContentAudio, URL: "/tmp/audio.wav"}, ""},
		{"assistant output", llmprotocol.RoleAssistant, llmprotocol.Content{Kind: llmprotocol.ContentAudio, MediaType: "audio/wav", Data: "AAAA"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			snapshot := extractSemanticRequestSignals(&llmprotocol.Request{Messages: []llmprotocol.Message{{Role: tc.role, Content: []llmprotocol.Content{tc.content}}}})
			if snapshot.FirstAudio != tc.want {
				t.Fatalf("audio reference = %q, want %q", snapshot.FirstAudio, tc.want)
			}
		})
	}
}

func TestChatAudioWireReachesNeutralSignalSnapshot(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.OpenAIChatV1, `{"model":"audio-model","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}`)
	snapshot := extractSemanticRequestSignals(request)
	if snapshot.FirstAudio != "data:audio/wav;base64,UklGRg==" {
		t.Fatalf("wire audio did not reach routing: %q", snapshot.FirstAudio)
	}
}
