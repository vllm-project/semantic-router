package classification

import (
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A routing question, a rule that does not read its whole input, reads a
// bounded view of a long state: a text part over its share of the
// deployment's routing view keeps its head and its tail around a marker. A
// state within the view goes as it is. The guard rules read their whole
// input instead, in windows (classifier_signal_scan.go).
//
// Views and windows are sized in bytes. A configured token budget takes
// routingViewBytesPerToken bytes a token, about what prose takes; the
// model's own cap takes one byte a token, since a byte-level tokenizer gives
// no byte more than one token, so no view or window overruns the model
// however dense its text (JSON or base64 take far fewer bytes a token than
// prose).

// routingViewReserveTokens is the part of a model's question cap a view or a
// scan window leaves to the question itself and the model's prompt template.
const routingViewReserveTokens = 1024

// routingViewBytesPerToken turns a configured token budget into bytes.
const routingViewBytesPerToken = 4

// routingViewMarker stands for the middle a view leaves out of a part, and
// says about how many tokens it leaves out.
const routingViewMarker = "\n\n[... %d tokens omitted ...]\n\n"

// routingViewSnap is how far, in bytes, a cut moves to fall between words.
const routingViewSnap = 64

// routingViewBudget is the most bytes a routing question to deployment
// reads of one state, and the share of them its head takes.
func (c *Classifier) routingViewBudget(deployment string) (int, float64) {
	tokens, share := c.Config.ModelDeployments[deployment].RoutingViewTokens()
	return c.budgetBytes(deployment, tokens), share
}

// budgetBytes is a budget of tokens as bytes, at most the question cap of
// deployment's card less the reserve (half the cap for a very small one),
// counted a byte a token.
func (c *Classifier) budgetBytes(deployment string, tokens int) int {
	bytes := tokens * routingViewBytesPerToken
	card, known := c.deploymentCard(deployment)
	if !known {
		return bytes
	}
	if limit := card.QuestionTokens(); limit > 0 {
		bytes = min(bytes, max(limit-routingViewReserveTokens, limit/2))
	}
	return bytes
}

// deploymentCard is the card a deployment's questions compile against: a
// systemone deployment's remote card, else the model runtime's.
func (c *Classifier) deploymentCard(deployment string) (modelservice.ModelCard, bool) {
	if c.Config.IsSystemOneDeployment(deployment) {
		if remote := c.systemOne(); remote != nil {
			return remote.Card(deployment)
		}
		return modelservice.ModelCard{}, false
	}
	return c.decisionTaskCard(deployment)
}

// stateBytes is the length of a request's text: its state, or its parts.
func stateBytes(request modelservice.Request) int {
	total := len(request.State)
	for _, text := range request.Parts {
		total += len(text)
	}
	return total
}

// boundedState is request with its text within budget bytes: as it is when
// it fits, else with each part over its share of the budget (the share that
// leaves the parts within it whole) reduced to its head and tail, the head
// taking headShare of it. Images and videos are kept.
func boundedState(request modelservice.Request, budget int, headShare float64) modelservice.Request {
	if budget <= 0 || stateBytes(request) <= budget {
		return request
	}
	if request.Parts == nil {
		request.State = routingView(request.State, budget, headShare)
		return request
	}
	names := make([]string, 0, len(request.Parts))
	for name := range request.Parts {
		names = append(names, name)
	}
	slices.Sort(names)
	sizes := make([]int, len(names))
	for index, name := range names {
		sizes[index] = len(request.Parts[name])
	}
	shares := fairShares(sizes, budget)
	parts := make(map[string]string, len(names))
	for index, name := range names {
		parts[name] = request.Parts[name]
		if sizes[index] > shares[index] {
			parts[name] = routingView(parts[name], shares[index], headShare)
		}
	}
	request.Parts = parts
	return request
}

// fairShares divides limit among parts of the given sizes: a part within an
// even share of what the smaller parts leave keeps its size, and the parts
// over it share the rest evenly.
func fairShares(sizes []int, limit int) []int {
	order := make([]int, len(sizes))
	for index := range order {
		order[index] = index
	}
	slices.SortStableFunc(order, func(a, b int) int { return sizes[a] - sizes[b] })
	shares := make([]int, len(sizes))
	left := limit
	for rank, index := range order {
		share := left / (len(order) - rank)
		if sizes[index] < share {
			share = sizes[index]
		}
		shares[index] = share
		left -= share
	}
	return shares
}

// routingView is text within limit bytes: as it is when it fits, else its
// head (headShare of what the marker leaves) and its tail, cut between
// characters, and between words when a word boundary is near.
func routingView(text string, limit int, headShare float64) string {
	if len(text) <= limit {
		return text
	}
	// The marker never says more tokens than the text has bytes.
	keep := max(limit-len(fmt.Sprintf(routingViewMarker, len(text))), 0)
	head := int(float64(keep) * min(max(headShare, 0), 1))
	headEnd := snapBack(text, runeStartBefore(text, head))
	tailStart := max(snapForward(text, runeStartAfter(text, len(text)-(keep-head))), headEnd)
	omitted := min(signalTextUnits(text[headEnd:tailStart])/4, len(text))
	return text[:headEnd] + fmt.Sprintf(routingViewMarker, omitted) + text[tailStart:]
}

// runeStartBefore is index, moved back to the start of the character it
// falls inside; within invalid UTF-8 it stays.
func runeStartBefore(text string, index int) int {
	if index <= 0 || index >= len(text) {
		return index
	}
	for at := index; at >= 0 && index-at < utf8.UTFMax; at-- {
		if utf8.RuneStart(text[at]) {
			return at
		}
	}
	return index
}

// runeStartAfter is index, moved on past the character it falls inside;
// within invalid UTF-8 it stays.
func runeStartAfter(text string, index int) int {
	if index <= 0 || index >= len(text) {
		return index
	}
	for at := index; at-index < utf8.UTFMax; at++ {
		if at == len(text) || utf8.RuneStart(text[at]) {
			return at
		}
	}
	return index
}

// signalTextUnits is the quarter-token estimate of text, signalUnitPrefix's
// total, read without allocating. It only says about how many tokens a view
// leaves out; views and windows are sized in bytes.
func signalTextUnits(text string) int {
	total, inWord, wordHasDigit := 0, false, false
	for index := 0; index < len(text); {
		r, size := utf8.DecodeRuneInString(text[index:])
		var units int
		units, inWord, wordHasDigit = signalRuneUnits(r, inWord, wordHasDigit)
		total += units
		index += size
	}
	return total
}

const routingViewSpaces = " \t\r\n"

// snapBack moves a head's end that falls inside a word back to the
// whitespace before that word, when it is within routingViewSnap bytes.
func snapBack(text string, end int) int {
	if end <= 0 || end >= len(text) || strings.IndexByte(routingViewSpaces, text[end]) >= 0 || strings.IndexByte(routingViewSpaces, text[end-1]) >= 0 {
		return end
	}
	from := max(0, end-routingViewSnap)
	if space := strings.LastIndexAny(text[from:end], routingViewSpaces); space >= 0 {
		return from + space
	}
	return end
}

// snapForward moves a tail's start that falls inside a word past the
// whitespace after that word, when it is within routingViewSnap bytes.
func snapForward(text string, start int) int {
	if start <= 0 || start >= len(text) || strings.IndexByte(routingViewSpaces, text[start-1]) >= 0 {
		return start
	}
	if space := strings.IndexAny(text[start:min(len(text), start+routingViewSnap)], routingViewSpaces); space >= 0 {
		return start + space + 1
	}
	return start
}
