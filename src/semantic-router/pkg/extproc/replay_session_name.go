package extproc

import (
	"encoding/json"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
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
	maxSessionNameRunes     = 120
)

// observeSessionTitleRequest recognises the side request a harness sends to
// name its session from what the harness marks: a Claude Code auxiliary
// request asking for a JSON object of one title field, or a Codex thread
// whose turn metadata says thread_title. OpenCode marks nothing, so its title
// call is the one exception read from its prompt: a tool-less call that opens
// with OpenCode's built-in title instructions.
func observeSessionTitleRequest(ctx *RequestContext, request *llmprotocol.Request) {
	ctx.SessionTitleRequest = ctx.AgentTurn.ThreadSource == codexThreadSourceTitle ||
		(strings.EqualFold(headerValueCI(ctx, headers.XClaudeCodeRequestClass), claudeCodeAuxiliaryClass) &&
			asksForTitleOnly(request.OutputFormat)) ||
		isOpenCodeTitleRequest(ctx, request)
}

// replaySideRequestKind is the kind of a request that is no turn of its
// agent, as the client marks it: a check a tool sends, or a Claude Code
// auxiliary or compaction request.
func replaySideRequestKind(ctx *RequestContext) string {
	if headerValueCI(ctx, headers.VSRProbe) != "" {
		return store.RequestKindProbe
	}
	switch class := strings.ToLower(headerValueCI(ctx, headers.XClaudeCodeRequestClass)); class {
	case store.RequestKindAuxiliary, store.RequestKindCompaction:
		return class
	}
	return ""
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

// observeSessionTitle reads the name a title request answered with, a JSON
// object of one title field: Claude Code's, or that of a Codex title thread
// matched to its run, which may answer with the bare line instead. OpenCode
// asks for the title alone, as plain text on one line.
func observeSessionTitle(ctx *RequestContext, response *llmprotocol.Response) {
	if ctx == nil || !ctx.SessionTitleRequest || response == nil {
		return
	}
	if isOpenCode(ctx) {
		title, _, _ := strings.Cut(strings.TrimSpace(extractSemanticAssistantResponseText(response)), "\n")
		ctx.sessionName = boundedSessionName(title)
		return
	}
	codex := ctx.AgentTurn.ThreadSource == codexThreadSourceTitle
	if codex && ctx.titleSession == "" {
		return
	}
	content := strings.TrimSpace(semanticAssistantContent(response))
	var answer struct {
		Title string `json:"title"`
	}
	switch {
	case json.Unmarshal([]byte(content), &answer) == nil:
		ctx.sessionName = boundedSessionName(answer.Title)
	case codex:
		line, _, _ := strings.Cut(content, "\n")
		ctx.sessionName = boundedSessionName(strings.Trim(line, "\"' "))
	}
}

// settleReplaySessionName writes the name a title request answered with
// before its record ends, where records keep session names: whatever the
// prompt capture, and never on a masking route.
func settleReplaySessionName(ctx *RequestContext, recorder *routerreplay.Recorder, state string) {
	if ctx.sessionName == "" || state != routerreplay.LifecycleCompleted ||
		!ctx.RouterReplayPluginConfig.KeepsSessionName() || replaySuppressesBody(ctx) {
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
// pair names the session. The name is kept where records keep session names,
// whatever the prompt capture, and never on a masking route.
func replaySessionNaming(ctx *RequestContext) (kind string, name string) {
	if ctx.SessionTitleRequest {
		kind = store.RequestKindSessionTitle
	}
	if ctx.SemanticRequest == nil || requestHarness(ctx) != store.HarnessClaudeCode ||
		!ctx.RouterReplayPluginConfig.KeepsSessionName() || replaySuppressesBody(ctx) {
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

// applyReplayFirstPromptName names a harness run that no title or /rename
// named by its main agent's first prompt, shortened: a headless run, such as
// claude -p or codex exec, gets no title. Only the first turn of a harness
// run's main agent, no side request or Looper hop, and only where records
// keep session names. The record marks the name as the prompt's, so that a
// title or /rename names the run over it, sent before or after.
func applyReplayFirstPromptName(record *routerreplay.RoutingRecord, ctx *RequestContext, prompt string) {
	if record.SessionName != "" || record.RequestKind != "" || record.ParentRequestID != "" || record.SessionID == "" ||
		record.Agent == nil || record.Agent.Harness == "" || record.Agent.ID != "" ||
		!ctx.RouterReplayPluginConfig.KeepsSessionName() || replaySuppressesBody(ctx) || !replayFirstTurn(ctx.SemanticRequest) {
		return
	}
	record.SessionName = boundedOneLine(looper.WithoutSystemReminders(prompt), maxSessionNameRunes)
	record.SessionNameFromPrompt = record.SessionName != ""
	if record.Agent.Harness == store.HarnessCodex {
		replayTitleIndex.remember(replayTitleKey(ctx, prompt), record.SessionID)
	}
}

// replayFirstTurn reports a request that carries no answer of the model yet:
// a conversation's first turn.
func replayFirstTurn(request *llmprotocol.Request) bool {
	if request == nil {
		return false
	}
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleAssistant {
			return false
		}
	}
	return true
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
