package extproc

import (
	"encoding/json"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	// claudeCodeAuxiliaryClass is the request class Claude Code gives its
	// side requests: session titles, classifiers and summaries.
	claudeCodeAuxiliaryClass = "auxiliary"
	// codexThreadSourceTitle is the thread_source of the thread Codex starts
	// to title another one.
	codexThreadSourceTitle = "thread_title"
	// Claude Code echoes a /rename and its output into the next request.
	claudeCodeRenameCommand = "<command-name>/rename</command-name>"
	claudeCodeRenameOutput  = "<local-command-stdout>Session renamed to: "
	claudeCodeCommandOutput = "</local-command-stdout>"
	maxSessionNameRunes     = 200
)

// observeSessionTitleRequest recognises the side request a harness sends to
// name its session from what the harness marks, never from its prompt: a
// Claude Code auxiliary request asking for a JSON object of one title field,
// or a Codex thread whose turn metadata says thread_title.
func observeSessionTitleRequest(ctx *RequestContext, request *llmprotocol.Request) {
	ctx.SessionTitleRequest = ctx.AgentTurn.ThreadSource == codexThreadSourceTitle ||
		(strings.EqualFold(headerValueCI(ctx, headers.XClaudeCodeRequestClass), claudeCodeAuxiliaryClass) &&
			asksForTitleOnly(request.OutputFormat))
}

func asksForTitleOnly(format llmprotocol.OutputFormat) bool {
	if format.Kind != llmprotocol.OutputJSONSchema {
		return false
	}
	var schema struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
		Required []string `json:"required"`
	}
	if json.Unmarshal(format.Schema, &schema) != nil {
		return false
	}
	title, ok := schema.Properties["title"]
	return ok && len(schema.Properties) == 1 && title.Type == "string" && slices.Equal(schema.Required, []string{"title"})
}

// observeSessionTitle reads the name a title request answered with. A Codex
// title thread carries no field naming the thread it titles, so its answer
// names no run Replay knows.
func observeSessionTitle(ctx *RequestContext, response *llmprotocol.Response) {
	if ctx == nil || !ctx.SessionTitleRequest || ctx.AgentTurn.ThreadSource == codexThreadSourceTitle || response == nil {
		return
	}
	var answer struct {
		Title string `json:"title"`
	}
	if json.Unmarshal([]byte(semanticAssistantContent(response)), &answer) == nil {
		ctx.sessionName = boundedSessionName(answer.Title)
	}
}

// settleReplaySessionName writes the name a title request answered with
// before its record ends. The name is response text, kept only where Replay
// keeps the answer.
func settleReplaySessionName(ctx *RequestContext, recorder *routerreplay.Recorder, state string) {
	if ctx.sessionName == "" || state != routerreplay.LifecycleCompleted ||
		!replayResponseContentAllowed(ctx) || !ctx.RouterReplayPluginConfig.CapturesResponseText() {
		return
	}
	if err := recorder.UpdateSessionName(ctx.RouterReplayID, ctx.sessionName); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_session_name_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

// replaySessionNaming is the request kind and session name a new record
// starts with. A Claude Code /rename reaches the gateway only as the
// command and its output in the next request's history, so the last such
// pair names the session. That is request text, kept as Replay keeps the
// request's tools: under raw capture with request text captured, and never
// on a masking route.
func replaySessionNaming(ctx *RequestContext) (kind string, name string) {
	if ctx.SessionTitleRequest {
		kind = store.RequestKindSessionTitle
	}
	policy := ctx.RouterReplayPluginConfig
	if ctx.SemanticRequest == nil || requestHarness(ctx) != store.HarnessClaudeCode || policy == nil ||
		!policy.CapturesRawContent() || !policy.CapturesRequestText() || replaySuppressesBody(ctx) {
		return kind, ""
	}
	for _, message := range ctx.SemanticRequest.Messages {
		if message.Role != llmprotocol.RoleUser {
			continue
		}
		if renamed := claudeCodeRename(message); renamed != "" {
			name = renamed
		}
	}
	return kind, boundedSessionName(name)
}

// claudeCodeRename is the name a /rename echo in message gave, the output
// that follows the command.
func claudeCodeRename(message llmprotocol.Message) string {
	name := ""
	for index := 1; index < len(message.Content); index++ {
		command, output := message.Content[index-1], message.Content[index]
		if command.Kind != llmprotocol.ContentText || output.Kind != llmprotocol.ContentText ||
			!strings.HasPrefix(strings.TrimSpace(command.Text), claudeCodeRenameCommand) {
			continue
		}
		text, prefixed := strings.CutPrefix(strings.TrimSpace(output.Text), claudeCodeRenameOutput)
		if renamed, suffixed := strings.CutSuffix(text, claudeCodeCommandOutput); prefixed && suffixed {
			name = renamed
		}
	}
	return name
}

func boundedSessionName(name string) string {
	name = strings.TrimSpace(name)
	if runes := []rune(name); len(runes) > maxSessionNameRunes {
		name = string(runes[:maxSessionNameRunes])
	}
	return name
}
