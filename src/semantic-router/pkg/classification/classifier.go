package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Classifier handles text classification, model selection, and jailbreak detection functionality
type Classifier struct {
	// Dependencies - In-tree classifiers
	categoryInitializer         CategoryInitializer
	categoryInference           CategoryInference
	jailbreakInitializer        JailbreakInitializer
	jailbreakInference          JailbreakInference
	piiInitializer              PIIInitializer
	piiInference                PIIInference
	keywordClassifier           *KeywordClassifier
	keywordEmbeddingInitializer EmbeddingClassifierInitializer
	keywordEmbeddingClassifier  *EmbeddingClassifier

	// Dependencies - MCP-based classifiers
	mcpCategoryInitializer MCPCategoryInitializer
	mcpCategoryInference   MCPCategoryInference

	// Hallucination mitigation classifiers
	factCheckClassifier   *FactCheckClassifier
	hallucinationDetector *HallucinationDetector
	feedbackDetector      *FeedbackDetector
	reaskClassifier       *ReaskClassifier

	// Preference classifier for route matching via external LLM
	preferenceClassifier *PreferenceClassifier

	// Language classifier
	languageClassifier *LanguageClassifier

	// Context classifier for token count-based routing
	contextClassifier *ContextClassifier
	// tokenCalibrator learns provider-specific prompt token ratios for context routing.
	tokenCalibrator *CalibratedTokenCounter

	// Structure classifier for request-shape routing signals
	structureClassifier *StructureClassifier

	// Complexity classifier for complexity-based routing using embedding similarity
	complexityClassifier *ComplexityClassifier

	// Event classifier for event-driven request routing
	eventClassifier *EventClassifier

	// Contrastive jailbreak classifiers keyed by rule name.
	// Only populated for JailbreakRules with Method == "contrastive".
	contrastiveJailbreakClassifiers map[string]*ContrastiveJailbreakClassifier

	// Authz classifier for user-level authorization signal classification
	authzClassifier *AuthzClassifier

	// Knowledge-base classifiers keyed by configured KB name.
	kbClassifiers map[string]*KnowledgeBaseClassifier

	// Identity header names resolved from authz.identity config (or defaults).
	// Used by EvaluateAllSignalsWithHeaders to read user identity from requests.
	authzUserIDHeader     string
	authzUserGroupsHeader string
	// authzFailOpen: cfg.Authz.FailOpen; see applyAuthzFailOpenOnClassifyError.
	authzFailOpen bool

	Config           *config.RouterConfig
	CategoryMapping  *CategoryMapping
	PIIMapping       *PIIMapping
	JailbreakMapping *JailbreakMapping

	// Category name mapping layer to support generic categories in config
	// Maps MMLU-Pro category names -> generic category names (as defined in config.Categories)
	MMLUToGeneric map[string]string
	// Maps generic category names -> MMLU-Pro category names
	GenericToMMLU map[string][]string
}

type option func(*Classifier)

func withCategory(categoryMapping *CategoryMapping, categoryInitializer CategoryInitializer, categoryInference CategoryInference) option {
	return func(c *Classifier) {
		c.CategoryMapping = categoryMapping
		c.categoryInitializer = categoryInitializer
		c.categoryInference = categoryInference
	}
}

func withJailbreak(jailbreakMapping *JailbreakMapping, jailbreakInitializer JailbreakInitializer, jailbreakInference JailbreakInference) option {
	return func(c *Classifier) {
		c.JailbreakMapping = jailbreakMapping
		c.jailbreakInitializer = jailbreakInitializer
		c.jailbreakInference = jailbreakInference
	}
}

func withPII(piiMapping *PIIMapping, piiInitializer PIIInitializer, piiInference PIIInference) option {
	return func(c *Classifier) {
		c.PIIMapping = piiMapping
		c.piiInitializer = piiInitializer
		c.piiInference = piiInference
	}
}

func withKeywordClassifier(keywordClassifier *KeywordClassifier) option {
	return func(c *Classifier) {
		c.keywordClassifier = keywordClassifier
	}
}

func withKeywordEmbeddingClassifier(keywordEmbeddingInitializer EmbeddingClassifierInitializer, keywordEmbeddingClassifier *EmbeddingClassifier) option {
	return func(c *Classifier) {
		c.keywordEmbeddingInitializer = keywordEmbeddingInitializer
		c.keywordEmbeddingClassifier = keywordEmbeddingClassifier
	}
}

func withReaskClassifier(reaskClassifier *ReaskClassifier) option {
	return func(c *Classifier) {
		c.reaskClassifier = reaskClassifier
	}
}

func withKBClassifiers(classifiers map[string]*KnowledgeBaseClassifier) option {
	return func(c *Classifier) {
		c.kbClassifiers = classifiers
	}
}

func withStructureClassifier(structureClassifier *StructureClassifier) option {
	return func(c *Classifier) {
		c.structureClassifier = structureClassifier
	}
}

func withComplexityClassifier(complexityClassifier *ComplexityClassifier) option {
	return func(c *Classifier) {
		c.complexityClassifier = complexityClassifier
	}
}

func withEventClassifier(eventClassifier *EventClassifier) option {
	return func(c *Classifier) {
		c.eventClassifier = eventClassifier
	}
}

func withContrastiveJailbreakClassifiers(classifiers map[string]*ContrastiveJailbreakClassifier) option {
	return func(c *Classifier) {
		c.contrastiveJailbreakClassifiers = classifiers
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

	classifier := &Classifier{Config: cfg}

	// Resolve identity header names from authz.identity config (or defaults).
	classifier.authzUserIDHeader = cfg.Authz.Identity.GetUserIDHeader()
	classifier.authzUserGroupsHeader = cfg.Authz.Identity.GetUserGroupsHeader()
	classifier.authzFailOpen = cfg.Authz.FailOpen

	for _, option := range options {
		option(classifier)
	}

	// Build category name mappings to support generic categories in config
	classifier.buildCategoryNameMappings()

	return classifier, nil
}
