package classification

// UsesLocalNativeEmbeddings reports whether a signal evaluation can reach a
// process-local embedding backend for the current request shape. Runtime
// component ownership and the effective backend are more trustworthy than the
// requested config alone. EMBEDDING_BACKEND_OVERRIDE has the same priority as
// EmbeddingClassifier.computeEmbedding, including when a remote-configured
// classifier still owns a provider instance. Image rules remain local because
// the remote text provider does not implement the multimodal image capability.
func (c *Classifier) UsesLocalNativeEmbeddings(hasImage bool) bool {
	if c == nil {
		return false
	}
	return c.keywordRulesUseLocalNativeEmbeddings(hasImage) ||
		c.reaskUsesLocalNativeEmbeddings() ||
		c.complexityUsesLocalNativeEmbeddings() ||
		c.preferenceUsesLocalNativeEmbeddings() ||
		c.jailbreakUsesLocalNativeEmbeddings() ||
		c.knowledgeBaseUsesLocalNativeEmbeddings()
}

func (c *Classifier) keywordRulesUseLocalNativeEmbeddings(hasImage bool) bool {
	classifier := c.keywordEmbeddingClassifier
	if classifier == nil {
		return false
	}
	usesLocalText := classifier.usesLocalNativeTextBackend() &&
		len(classifier.rulesByModality["text"]) > 0
	usesLocalImage := hasImage && len(classifier.rulesByModality["image"]) > 0
	return usesLocalText || usesLocalImage
}

func (c *Classifier) reaskUsesLocalNativeEmbeddings() bool {
	classifier := c.reaskClassifier
	return classifier != nil &&
		textEmbeddingUsesLocalNativeBackend(classifier.backend, classifier.provider)
}

func (c *Classifier) complexityUsesLocalNativeEmbeddings() bool {
	classifier := c.complexityClassifier
	if classifier == nil {
		return false
	}
	// Image-candidate complexity always computes a local multimodal text
	// embedding, and additionally a local image embedding when one is present.
	return classifier.hasImageCandidates ||
		textEmbeddingUsesLocalNativeBackend(classifier.backend, classifier.provider)
}

func (c *Classifier) preferenceUsesLocalNativeEmbeddings() bool {
	classifier := c.preferenceClassifier
	return classifier != nil && classifier.useContrastive && classifier.contrastive != nil &&
		textEmbeddingUsesLocalNativeBackend(classifier.contrastive.backend, classifier.contrastive.provider)
}

func (c *Classifier) jailbreakUsesLocalNativeEmbeddings() bool {
	for _, classifier := range c.contrastiveJailbreakClassifiers {
		if classifier != nil && textEmbeddingUsesLocalNativeBackend(classifier.backend, classifier.provider) {
			return true
		}
	}
	return false
}

func (c *Classifier) knowledgeBaseUsesLocalNativeEmbeddings() bool {
	for _, classifier := range c.kbClassifiers {
		if classifier != nil && textEmbeddingUsesLocalNativeBackend(classifier.backend, classifier.provider) {
			return true
		}
	}
	return false
}

func (c *EmbeddingClassifier) usesLocalNativeTextBackend() bool {
	if c == nil {
		return false
	}
	return isLocalNativeEmbeddingBackend(effectiveTextEmbeddingBackend(c.backend, c.provider))
}
