package topiccontinuity

import "strings"

// The v1 lexicons. Any change here changes evaluator behavior and must bump
// EvaluatorVersion. Phrases are matched as contiguous token sequences in the
// normalized phrase views (see normalize.go).

var strongReferencePhrases = phraseList(
	"above", "earlier", "previous", "previously", "as before", "like before", "same as",
	"as you said", "you said", "you mentioned", "you suggested", "your answer", "your code",
	"your example", "your suggestion", "that function", "that file", "that code", "that one",
	"this one", "the last one", "instead", "go back to", "as mentioned", "mentioned earlier",
)

// weakReferenceTokens are matched only within the first three tokens of the
// first live segment, after leading acknowledgements are stripped.
var weakReferenceTokens = tokenSet(
	"also", "now", "then", "and", "but", "next", "so", "it", "its", "this", "that", "these",
	"those", "they", "them",
)

var acknowledgementPhrases = phraseList(
	"thanks", "thank you", "thx", "ok", "okay", "great", "good", "done", "perfect", "got it",
	"nice", "cool", "sounds good", "awesome", "yes", "yep", "sure",
)

var changeMarkerPhrases = phraseList(
	"new question", "unrelated", "different topic", "different question", "change of topic",
	"changing topics", "switching topics", "switch topics", "on another note", "separate question",
	"completely different", "new topic", "something else",
)

var negationTokens = tokenSet(
	"not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
	"without", "hardly", "nor",
)

var stopwords = tokenSet(
	"a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at", "by",
	"for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this",
	"that", "these", "those", "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
	"their", "do", "does", "did", "can", "could", "would", "should", "will", "just", "please",
	"how", "what", "why", "when", "where", "which", "who", "not", "no",
)

const (
	weakReferenceWindow = 3
	changeLeadingWindow = 8
	negationLookback    = 3
	maxAckTokens        = 4
)

// phraseList splits each phrase into its tokens, ordered longest first so
// greedy matching prefers multi-token phrases.
func phraseList(phrases ...string) [][]string {
	out := make([][]string, 0, len(phrases))
	for _, phrase := range phrases {
		out = append(out, strings.Fields(phrase))
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && len(out[j]) > len(out[j-1]); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

func tokenSet(tokens ...string) map[string]struct{} {
	out := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		out[token] = struct{}{}
	}
	return out
}
