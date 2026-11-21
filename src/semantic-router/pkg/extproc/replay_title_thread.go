package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	// codexTitlePromptMarker precedes the first prompt in the request Codex
	// 0.166.0 sends on its title thread.
	codexTitlePromptMarker = "\nUser prompt:\n"
	// A title thread follows its run's first turn within seconds; a match
	// older than this, or past the index's bound, is dropped.
	replayTitleWindow   = 2 * time.Minute
	replayTitleCapacity = 1024
)

// replayTitleIndex remembers, for a short while and in this process only, the
// run each Codex first prompt started, keyed by a hash of the client and the
// prompt, so that the title thread Codex starts for it can be filed under
// that run. No prompt text is kept.
var replayTitleIndex = &replayTitleLinks{runs: make(map[string]replayTitleRun)}

type replayTitleRun struct {
	session string
	at      time.Time
}

type replayTitleLinks struct {
	mu    sync.Mutex
	runs  map[string]replayTitleRun
	order []string
}

func (l *replayTitleLinks) remember(key, session string) {
	if key == "" || session == "" {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, known := l.runs[key]; !known {
		l.order = append(l.order, key)
	}
	l.runs[key] = replayTitleRun{session: session, at: time.Now()}
	for len(l.order) > replayTitleCapacity {
		delete(l.runs, l.order[0])
		l.order = l.order[1:]
	}
}

func (l *replayTitleLinks) find(key string) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	run, ok := l.runs[key]
	if !ok || time.Since(run.at) > replayTitleWindow {
		return ""
	}
	return run.session
}

// replayTitleKey hashes a first prompt with the client that sent it: the
// gateway's authenticated user, the ChatGPT account, and the credential the
// client presented, so that one client's title never names another's run.
func replayTitleKey(ctx *RequestContext, prompt string) string {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return ""
	}
	sum := sha256.New()
	for _, part := range []string{
		ctx.AuthenticatedPrincipal, headerValueCI(ctx, "chatgpt-account-id"), headerValueCI(ctx, headers.Authorization), prompt,
	} {
		sum.Write([]byte(part))
		sum.Write([]byte{0})
	}
	return hex.EncodeToString(sum.Sum(nil))
}

// linkReplayTitleThread files a Codex title thread under the run whose first
// prompt it titles. Codex starts the thread under a session of its own, and
// its request carries the first prompt after a fixed marker: a match on the
// prompt from the same client within a short window, where records keep
// session names and never on a masking route, makes it that run's title
// request.
func linkReplayTitleThread(record *routerreplay.RoutingRecord, ctx *RequestContext, prompt string) {
	if !ctx.SessionTitleRequest || ctx.AgentTurn.ThreadSource != codexThreadSourceTitle ||
		!ctx.RouterReplayPluginConfig.KeepsSessionName() || replaySuppressesBody(ctx) {
		return
	}
	_, first, found := strings.Cut(prompt, codexTitlePromptMarker)
	if !found {
		return
	}
	if session := replayTitleIndex.find(replayTitleKey(ctx, first)); session != "" {
		record.SessionID, ctx.titleSession = session, session
		record.Agent = &store.AgentIdentity{Harness: store.HarnessCodex}
	}
}
