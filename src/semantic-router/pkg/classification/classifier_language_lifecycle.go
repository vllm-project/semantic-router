package classification

import "fmt"

// IsLanguageEnabled checks if language classification is enabled.
func (c *Classifier) IsLanguageEnabled() bool {
	return len(c.Config.LanguageRules) > 0 && c.languageClassifier != nil
}

// initializeLanguageClassifier initializes the language classifier.
func (c *Classifier) initializeLanguageClassifier() error {
	if len(c.Config.LanguageRules) == 0 {
		return nil
	}

	classifier, err := NewLanguageClassifier(c.Config.LanguageRules)
	if err != nil {
		return fmt.Errorf("failed to create language classifier: %w", err)
	}

	c.languageClassifier = classifier
	return nil
}

// GetLanguageClassifier returns the language classifier instance.
func (c *Classifier) GetLanguageClassifier() *LanguageClassifier {
	return c.languageClassifier
}
