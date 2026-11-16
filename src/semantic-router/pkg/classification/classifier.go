package classification

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Classifier evaluates a recipe's signals.
type Classifier struct {
	closeOnce         sync.Once
	closeErr          error
	embeddingSet      *embedding.Set
	ownsEmbeddingSet  bool
	models            *classifierModelRuntime
	keywordClassifier *KeywordClassifier

	// Language classifier
	languageClassifier *LanguageClassifier

	// Context classifier for token count-based routing
	contextClassifier *ContextClassifier

	// decisionDecider answers systemone and application signals; nil uses
	// the process-wide model runtime manager.
	decisionDecider modelservice.Decider
	// systemOneDecider answers the signals asked of systemone deployments;
	// nil uses the signal runtime's remote System One clients.
	systemOneDecider modelservice.Remote
	// Immutable cards captured while preparing this classifier generation.
	decisionCards map[string]modelservice.ModelCard
	// applicationRules are the configuration's application signal rules,
	// compiled into the systemone rules they ask.
	applicationRules []config.ApplicationRule
	// windowPaces are the paces its deployments read guard windows at, which
	// say whether a scan that fell behind its deadline reads the rest of its
	// windows in time at a larger size.
	windowPaces windowPaces
	// overloadGates pace the requests that wait for an overloaded decision
	// runtime to have capacity (on_signal_overload).
	overloadGates overloadGates
	// tokenCalibrator learns provider-specific prompt token ratios for context routing.
	tokenCalibrator *CalibratedTokenCounter

	// Authz classifier for user-level authorization signal classification
	authzClassifier *AuthzClassifier

	// Identity header names resolved from authz.identity config (or defaults).
	// Used by EvaluateAllSignalsWithHeaders to read user identity from requests.
	authzUserIDHeader     string
	authzUserGroupsHeader string
	// authzFailOpen: cfg.Authz.FailOpen; see applyAuthzFailOpenOnClassifyError.
	authzFailOpen bool

	Config *config.RouterConfig
}

type option func(*Classifier)

func withKeywordClassifier(keywordClassifier *KeywordClassifier) option {
	return func(c *Classifier) {
		c.keywordClassifier = keywordClassifier
	}
}

func withAuthzClassifier(authzClassifier *AuthzClassifier) option {
	return func(c *Classifier) {
		c.authzClassifier = authzClassifier
	}
}

// newClassifierWithOptions creates a new classifier with the given options
func newClassifierWithOptions(cfg *config.RouterConfig, options ...option) (*Classifier, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	classifier := &Classifier{Config: cfg, applicationRules: cfg.ApplicationRules()}

	// Resolve identity header names from authz.identity config (or defaults).
	classifier.authzUserIDHeader = cfg.Authz.Identity.GetUserIDHeader()
	classifier.authzUserGroupsHeader = cfg.Authz.Identity.GetUserGroupsHeader()
	classifier.authzFailOpen = cfg.Authz.FailOpen

	for _, option := range options {
		option(classifier)
	}

	return classifier, nil
}
