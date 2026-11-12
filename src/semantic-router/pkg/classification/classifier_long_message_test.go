package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// NewDecisionModelTestClassifier is available only to external tests: the
// classifier of cfg, whose decision questions decider answers against the
// given cards, as in a generation that prepared them.
func NewDecisionModelTestClassifier(cfg *config.RouterConfig, decider modelservice.Decider, cards map[string]modelservice.ModelCard) (*Classifier, error) {
	classifier, err := newClassifierWithOptions(cfg)
	if err != nil {
		return nil, err
	}
	classifier.SetDecisionDecider(decider)
	classifier.decisionCards = cards
	return classifier, nil
}

// LongMessageTestText is available only to external tests: a benign made-up
// text of about tokens estimated tokens, as the Router estimates them, with
// inserts after the paragraphs their keys number, then ending.
func LongMessageTestText(tokens int, inserts map[int]string, ending string) string {
	return longChapterWith(tokens, inserts, ending)
}

// DenseTestText is available only to external tests: a made-up JSON export
// ("json") or base64 attachment ("base64") of about size bytes, then ending.
func DenseTestText(kind string, size int, ending string) string {
	return denseText(kind, size, ending)
}
