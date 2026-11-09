// Package routinginput applies global.router.routing_input: it decides which
// user messages are routing text and cuts configured blocks from that text.
package routinginput

import (
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Selector is one compiled routing_input configuration. A nil Selector
// selects every message and strips nothing.
type Selector struct {
	skip   []*regexp.Regexp
	blocks []config.RoutingInputStripBlock
}

// maxCachedSelectors bounds the cache across configuration reloads.
const maxCachedSelectors = 16

var cache = struct {
	sync.Mutex
	selectors map[string]*Selector
}{selectors: map[string]*Selector{}}

// For returns the selector for cfg, or nil when cfg changes nothing. Router
// generations with the same settings share one compiled selector. Validation
// rejects settings that do not compile; such settings select nothing here.
func For(cfg config.RoutingInputConfig) *Selector {
	if !cfg.Enabled() {
		return nil
	}
	key := cacheKey(cfg)
	cache.Lock()
	defer cache.Unlock()
	if selector, ok := cache.selectors[key]; ok {
		return selector
	}
	selector, err := Compile(cfg)
	if err != nil {
		return nil
	}
	if len(cache.selectors) >= maxCachedSelectors {
		cache.selectors = map[string]*Selector{}
	}
	cache.selectors[key] = selector
	return selector
}

// Compile compiles cfg without consulting the cache.
func Compile(cfg config.RoutingInputConfig) (*Selector, error) {
	selector := &Selector{blocks: append([]config.RoutingInputStripBlock(nil), cfg.StripBlocks...)}
	for _, pattern := range cfg.SkipUserMessagesMatching {
		compiled, err := regexp.Compile(pattern)
		if err != nil {
			return nil, err
		}
		selector.skip = append(selector.skip, compiled)
	}
	return selector, nil
}

func cacheKey(cfg config.RoutingInputConfig) string {
	var key strings.Builder
	write := func(value string) {
		key.WriteString(strconv.Itoa(len(value)))
		key.WriteByte(':')
		key.WriteString(value)
	}
	for _, pattern := range cfg.SkipUserMessagesMatching {
		write(pattern)
	}
	key.WriteByte('|')
	for _, block := range cfg.StripBlocks {
		write(block.Start)
		write(block.End)
	}
	return key.String()
}

// Outcome is what routing_input changed in one request. The removed text
// serves the safety signals and the response cache identity; never log it.
type Outcome struct {
	SkippedMessages int
	StrippedBlocks  int
	// MessageIndex is the index of the user message the routing text came
	// from: the last kept user message with routing text, or -1.
	MessageIndex int
	// Stripped holds the blocks cut from that message.
	Stripped []string
	// ExcludedCurrent holds Stripped and the text of every user message after
	// the routing message, which routing skipped or stripped to nothing.
	ExcludedCurrent []string
	// ExcludedPrior holds the user messages before the routing message as sent,
	// skipped ones included.
	ExcludedPrior []string
}

// Changed reports whether routing_input removed anything from the request.
func (o Outcome) Changed() bool {
	return o.SkippedMessages > 0 || o.StrippedBlocks > 0
}

// Walk applies a selector to the user messages of one request, in order.
type Walk struct {
	selector *Selector
	turns    []turn
}

// turn is one user message with text, as sent and as routed.
type turn struct {
	index    int
	text     string
	routed   string
	skipped  bool
	stripped []string
}

// Walk starts a request walk. The walk of a nil selector changes nothing.
func (s *Selector) Walk() *Walk {
	return &Walk{selector: s}
}

// UserMessage records one user message and returns its routing text and
// whether routing skips it. Only a message whose content is all text can be
// skipped, so its media and tool results keep reaching their signals.
func (w *Walk) UserMessage(index int, text string, textOnly bool) (string, bool) {
	if w == nil || w.selector == nil || text == "" {
		return text, false
	}
	current := turn{index: index, text: text}
	if textOnly && w.selector.SkipsMessage(text) {
		current.skipped = true
	} else {
		current.routed, current.stripped = w.selector.Strip(text)
	}
	w.turns = append(w.turns, current)
	return current.routed, current.skipped
}

// Outcome names the routing message and assigns the removed text: the blocks
// cut from the routing message and every user message after it are current
// text, the user messages before it are prior messages.
func (w *Walk) Outcome() Outcome {
	outcome := Outcome{MessageIndex: -1}
	if w == nil {
		return outcome
	}
	routing := -1
	for position, turn := range w.turns {
		if turn.skipped {
			outcome.SkippedMessages++
			continue
		}
		outcome.StrippedBlocks += len(turn.stripped)
		if turn.routed != "" {
			routing = position
		}
	}
	if routing >= 0 {
		outcome.MessageIndex = w.turns[routing].index
		outcome.Stripped = w.turns[routing].stripped
	}
	if !outcome.Changed() {
		return outcome
	}
	for position, turn := range w.turns {
		switch {
		case position < routing:
			outcome.ExcludedPrior = append(outcome.ExcludedPrior, turn.text)
		case position == routing:
			outcome.ExcludedCurrent = append(outcome.ExcludedCurrent, turn.stripped...)
		default:
			outcome.ExcludedCurrent = append(outcome.ExcludedCurrent, turn.text)
		}
	}
	return outcome
}

// SkipsMessage reports whether a user message with this text is left out of
// the routing text. Callers decide whether the message carries anything else.
func (s *Selector) SkipsMessage(text string) bool {
	if s == nil || text == "" {
		return false
	}
	for _, pattern := range s.skip {
		if pattern.MatchString(text) {
			return true
		}
	}
	return false
}

// Strip cuts the configured blocks from text in one left-to-right pass. The
// earliest start marker opens a block, the block listed first winning a tie,
// and the first matching end marker after it closes the block. A start marker
// without an end marker after it stays, and the pass continues after it. Strip
// returns the remaining text, trimmed when a block was cut, and the cut blocks
// with their markers.
func (s *Selector) Strip(text string) (string, []string) {
	if s == nil || len(s.blocks) == 0 || text == "" {
		return text, nil
	}
	const unknown = -2
	next := make([]int, len(s.blocks))
	for index := range next {
		next[index] = unknown
	}
	var kept strings.Builder
	var removed []string
	keptFrom, position := 0, 0
	for {
		chosen, start := -1, -1
		for index, block := range s.blocks {
			if next[index] == -1 {
				continue
			}
			if next[index] < position {
				offset := strings.Index(text[position:], block.Start)
				if offset < 0 {
					next[index] = -1
					continue
				}
				next[index] = position + offset
			}
			if start < 0 || next[index] < start {
				chosen, start = index, next[index]
			}
		}
		if chosen < 0 {
			break
		}
		block := s.blocks[chosen]
		body := start + len(block.Start)
		offset := strings.Index(text[body:], block.End)
		if offset < 0 {
			// No end marker follows, so no later start of this block closes either.
			next[chosen] = -1
			continue
		}
		end := body + offset + len(block.End)
		kept.WriteString(text[keptFrom:start])
		removed = append(removed, text[start:end])
		keptFrom, position = end, end
	}
	if len(removed) == 0 {
		return text, nil
	}
	kept.WriteString(text[keptFrom:])
	return strings.TrimSpace(kept.String()), removed
}
