package testcases

import (
	"encoding/json"
	"io/fs"
	"regexp"
	"strings"
	"testing"
)

var agentClientCaptureLeaks = map[string]*regexp.Regexp{
	"credential header": regexp.MustCompile(`(?i)"(authorization|x-api-key|api-key|cookie|x-claude-code-session-id)"\s*:`),
	"secret":            regexp.MustCompile(`sk-[A-Za-z0-9_-]{8,}|(?i)bearer\s+[A-Za-z0-9._-]{8,}`),
	"local path":        regexp.MustCompile(`/Users/|/home/|/private/|/var/folders/|[A-Za-z]:\\\\`),
	"email address":     regexp.MustCompile(`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`),
	"IP address":        regexp.MustCompile(`\b(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}\b`),
	"hostname":          regexp.MustCompile(`(?i)localhost|\.local\b|\.internal\b|\.corp\b|\.lan\b`),
}

func TestAgentClientCapturesAreSanitized(t *testing.T) {
	names, err := fs.Glob(agentClientCaptureFiles, "testdata/agent_clients/*.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		data, readErr := agentClientCaptureFiles.ReadFile(name)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if len(data) > 16<<10 {
			t.Errorf("%s is %d bytes; shorten it below 16 KiB", name, len(data))
		}
		for kind, pattern := range agentClientCaptureLeaks {
			if match := pattern.Find(data); match != nil {
				t.Errorf("%s contains a %s: %q", name, kind, match)
			}
		}
	}
}

func TestAgentClientCapturesCoverEveryBackend(t *testing.T) {
	captures, err := loadAgentClientCaptures()
	if err != nil {
		t.Fatal(err)
	}
	for _, backendFormat := range []string{"openai.chat.v1", "openai.responses.v1", "anthropic.messages.v1"} {
		covered := false
		for _, capture := range captures {
			if _, ok := capture.Backends[backendFormat]; ok {
				covered = true
			}
		}
		if !covered {
			t.Errorf("no agent-client capture replays against %s", backendFormat)
		}
	}
}

func TestAgentClientTurnMarksOnlyTheLastUserText(t *testing.T) {
	captures, err := loadAgentClientCaptures()
	if err != nil {
		t.Fatal(err)
	}
	for _, capture := range captures {
		protocol := agentClientProtocols[capture.Path]
		for _, stream := range []bool{true, false} {
			turn, turnErr := agentClientTurn(protocol, capture.Turns[0], "profile-model", stream, agentClientToolMarker)
			if turnErr != nil {
				t.Fatalf("%s: %v", capture.Name, turnErr)
			}
			text, _, textErr := protocol.userText(turn)
			encoded, encodeErr := json.Marshal(turn)
			if textErr != nil || encodeErr != nil || !strings.HasSuffix(text, "\n"+agentClientToolMarker) ||
				strings.Count(string(encoded), agentClientToolMarker) != 1 {
				t.Errorf("%s stream=%t: marker is not the only suffix of the last user text: %v %v", capture.Name, stream, textErr, encodeErr)
			}
			if turn["model"] != "profile-model" || turn["stream"] != stream {
				t.Errorf("%s stream=%t: model or stream not replaced: %v %v", capture.Name, stream, turn["model"], turn["stream"])
			}
			if _, ok := turn["stream_options"]; ok && !stream {
				t.Errorf("%s: buffered turn kept stream_options", capture.Name)
			}
		}
	}
}

func TestAgentClientUserTextShapes(t *testing.T) {
	tests := []struct {
		name     string
		protocol string
		turn     string
		want     string
	}{
		{
			name: "chat string content", protocol: "/v1/chat/completions", want: "second",
			turn: `{"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"ok"},{"role":"user","content":"second"}]}`,
		},
		{
			name: "messages text parts", protocol: "/v1/messages", want: "prompt",
			turn: `{"messages":[{"role":"user","content":[{"type":"text","text":"reminder"},{"type":"text","text":"prompt"}]},{"role":"system","content":"date"}]}`,
		},
		{
			name: "responses string input", protocol: "/v1/responses", want: "prompt",
			turn: `{"input":"prompt"}`,
		},
		{
			name: "responses input items", protocol: "/v1/responses", want: "prompt",
			turn: `{"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"rules"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"prompt"}]}]}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var turn map[string]any
			if err := json.Unmarshal([]byte(test.turn), &turn); err != nil {
				t.Fatal(err)
			}
			text, set, err := agentClientProtocols[test.protocol].userText(turn)
			if err != nil || text != test.want {
				t.Fatalf("userText = %q, %v; want %q", text, err, test.want)
			}
			set("marked")
			encoded, _ := json.Marshal(turn)
			if strings.Count(string(encoded), "marked") != 1 {
				t.Fatalf("setter did not replace exactly one text: %s", encoded)
			}
		})
	}
	if _, _, err := lastMessageUserText(map[string]any{"messages": []any{}}); err == nil {
		t.Fatal("a turn without user text was accepted")
	}
}
