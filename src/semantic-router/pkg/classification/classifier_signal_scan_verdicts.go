package classification

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// A session sends much of a guard rule's input again on every turn: the
// user turn a tool loop answers stays the request's current user turn until
// the user writes again, and a rule that reads the earlier turns reads them
// all again. A long input's windows would then be read again on every turn,
// for seconds a turn. For deployments without auth, client certificates or
// forwarded request headers, a window's answers depend on what its call sends
// (the window, the request's images and videos, the questions and the
// deployment's model), so the classifier keeps the answers of the request
// stage's windows for a while, and a window read again in that time takes
// them instead of a call: the rule still reads every window of its input,
// each once. A window's key holds a hash of all the text before it as well
// as its own, so a request finds a window's answers only when its input is
// the same up to the window's end: the answers kept for one conversation
// never tell another request anything, not even by how fast it is routed.

const (
	// windowVerdictEntries bounds the windows kept, the least recently read
	// leaving first.
	windowVerdictEntries = 16384
	// windowVerdictTTL is how long a window's answers stay after the last
	// request that read the window.
	windowVerdictTTL = 30 * time.Minute
	// windowVerdictBlock is the text a scan's prefix hashes advance by, so a
	// window's key hashes at most this much text of its own.
	windowVerdictBlock = 16 << 10
)

type windowVerdictKey [sha256.Size]byte

// windowVerdicts are the answers of the windows the classifier read, by
// window key.
type windowVerdicts struct {
	mu      sync.Mutex
	entries map[windowVerdictKey]*list.Element
	order   *list.List
}

type keptWindowEntry struct {
	key         windowVerdictKey
	response    modelservice.Response
	expires     time.Time
	sharedUntil time.Time
}

// get returns the answers kept for key, and keeps them windowVerdictTTL
// longer.
func (v *windowVerdicts) get(key windowVerdictKey) (modelservice.Response, bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	element, ok := v.entries[key]
	if !ok {
		return modelservice.Response{}, false
	}
	verdict := element.Value.(*keptWindowEntry)
	now := time.Now()
	if !now.Before(verdict.expires) {
		v.order.Remove(element)
		delete(v.entries, key)
		return modelservice.Response{}, false
	}
	verdict.expires = now.Add(windowVerdictTTL)
	v.order.MoveToFront(element)
	return cloneWindowResponse(verdict.response), true
}

// put keeps a window's answers.
func (v *windowVerdicts) put(key windowVerdictKey, response modelservice.Response) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.entries == nil {
		v.entries, v.order = map[windowVerdictKey]*list.Element{}, list.New()
	}
	expires := time.Now().Add(windowVerdictTTL)
	if element, ok := v.entries[key]; ok {
		verdict := element.Value.(*keptWindowEntry)
		verdict.response, verdict.expires = cloneWindowResponse(response), expires
		v.order.MoveToFront(element)
		return
	}
	v.entries[key] = v.order.PushFront(&keptWindowEntry{key: key, response: cloneWindowResponse(response), expires: expires})
	for v.order.Len() > windowVerdictEntries {
		oldest := v.order.Back()
		v.order.Remove(oldest)
		delete(v.entries, oldest.Value.(*keptWindowEntry).key)
	}
}

// keptWindow returns the answers an earlier request's read of the same input
// gave a window call, and marks the window reused.
func (c *Classifier) keptWindow(ctx context.Context, call *systemOneCall) (modelservice.Response, bool) {
	if call.verdict == nil {
		return modelservice.Response{}, false
	}
	response, ok := c.windowVerdicts.get(*call.verdict)
	if !ok {
		if c.sharedVerdicts != nil {
			metrics.RecordGuardVerdictStoreOp("local", "miss")
		}
		var expires time.Time
		response, expires, ok = c.sharedVerdicts.get(ctx, *call.verdict)
		if !ok {
			return modelservice.Response{}, false
		}
		if !keptAnswers(call, response) || !restoreWindowSpans(call, response) {
			metrics.RecordGuardVerdictStoreOp("get", "invalid")
			return modelservice.Response{}, false
		}
		metrics.RecordGuardVerdictStoreOp("get", "hit")
		c.windowVerdicts.put(*call.verdict, response)
		c.windowVerdicts.published(*call.verdict, expires)
	} else if c.sharedVerdicts != nil {
		metrics.RecordGuardVerdictStoreOp("local", "hit")
	}
	c.sharedVerdicts.touch(&c.windowVerdicts, *call.verdict, response)
	call.scan.mu.Lock()
	call.scan.reused[call.window] = true
	call.scan.mu.Unlock()
	metrics.RecordSignalScanWindow(call.deployment, "kept")
	return response, true
}

// keepWindow keeps the answers of a window call the deployment read, when
// they may be kept (keptAnswers).
func (c *Classifier) keepWindow(ctx context.Context, call *systemOneCall, response modelservice.Response, err error) {
	if err != nil {
		return
	}
	metrics.RecordSignalScanWindow(call.deployment, "read")
	if call.verdict != nil && keptAnswers(call, response) {
		c.windowVerdicts.put(*call.verdict, response)
		expires := c.sharedVerdicts.put(ctx, *call.verdict, response)
		c.windowVerdicts.published(*call.verdict, expires)
	}
}

// verdictKey keys a window of the scan in windowVerdicts, nil when the scan
// keeps none.
func (scan *guardScan) verdictKey(window scanWindow) *windowVerdictKey {
	if scan.verdicts == nil {
		return nil
	}
	key := scan.verdicts.key(scan.text, window.start, window.end)
	return &key
}

// keptAnswers reports whether a window call's response may be kept: every
// question it asked has an answer without an error that proves the model
// read the whole window.
func keptAnswers(call *systemOneCall, response modelservice.Response) bool {
	for _, question := range callQuestions(call.rules) {
		answer, ok := response.Answers[question.ID]
		if !ok || answer.Error != "" || answer.InputCoverage != "complete" {
			return false
		}
	}
	return len(response.Answers) > 0
}

func cloneWindowResponse(response modelservice.Response) modelservice.Response {
	answers := make(map[string]modelservice.Answer, len(response.Answers))
	for id, answer := range response.Answers {
		probabilities := make(map[string]float64, len(answer.Probabilities))
		for label, probability := range answer.Probabilities {
			probabilities[label] = probability
		}
		answer.Probabilities = probabilities
		answer.Selected = append([]string(nil), answer.Selected...)
		answer.Spans = append([]modelservice.Span(nil), answer.Spans...)
		answers[id] = answer
	}
	response.Answers = answers
	return response
}

// scanVerdictKeys keys the windows of one guard scan: a seed with what every
// window's call sends besides its text, and the hashes of the scan's text up
// to each windowVerdictBlock, which a window's key continues to its end. A
// scan of the response stage keeps nothing: it reads an answer that is new
// on every request.
type scanVerdictKeys struct {
	seed   [sha256.Size]byte
	blocks [][sha256.Size]byte
}

func (c *Classifier) newScanVerdictKeys(scan *guardScan) *scanVerdictKeys {
	for _, asked := range scan.rules {
		if asked.rule.EffectiveStage() != config.SystemOneStageRequest {
			return nil
		}
	}
	window := scan.windowState("")
	window.MaxTokens = c.Config.ModelDeployments[scan.deployment].ScanBudget()
	window.Questions = callQuestions(scan.rules)
	sent, err := json.Marshal(window)
	if err != nil {
		return nil
	}
	seed := sha256.New()
	model, ok := c.windowModelIdentity(scan.deployment)
	if !ok {
		return nil
	}
	for _, field := range append(model, string(sent)) {
		writeVerdictField(seed, field)
	}
	keys := &scanVerdictKeys{}
	seed.Sum(keys.seed[:0])
	chain := keys.seed
	for from := 0; from+windowVerdictBlock <= len(scan.text); from += windowVerdictBlock {
		block := sha256.New()
		_, _ = block.Write(chain[:])
		_, _ = io.WriteString(block, scan.text[from:from+windowVerdictBlock])
		block.Sum(chain[:0])
		keys.blocks = append(keys.blocks, chain)
	}
	return keys
}

// key is the key of the window of text from start to end: the hash of the
// text before end, the seed included, and the window's bounds.
func (keys *scanVerdictKeys) key(text string, start, end int) windowVerdictKey {
	chain, from := keys.seed, 0
	if blocks := end / windowVerdictBlock; blocks > 0 {
		chain, from = keys.blocks[blocks-1], blocks*windowVerdictBlock
	}
	hash := sha256.New()
	_, _ = hash.Write(chain[:])
	_, _ = io.WriteString(hash, text[from:end])
	var bounds [16]byte
	binary.LittleEndian.PutUint64(bounds[:8], uint64(start)) //nolint:gosec // a byte offset, never negative
	binary.LittleEndian.PutUint64(bounds[8:], uint64(end))   //nolint:gosec // a byte offset, never negative
	_, _ = hash.Write(bounds[:])
	var key windowVerdictKey
	hash.Sum(key[:0])
	return key
}

// writeVerdictField writes field to hash after its length, so no two lists
// of fields hash alike.
func writeVerdictField(hash io.Writer, field string) {
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
	_, _ = hash.Write(size[:])
	_, _ = io.WriteString(hash, field)
}
