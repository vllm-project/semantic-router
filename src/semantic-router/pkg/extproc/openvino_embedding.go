package extproc

import "strings"

func openvinoEmbeddingUsesModernBERT(modelType string) bool {
	switch strings.ToLower(strings.TrimSpace(modelType)) {
	case "mmbert", "modernbert":
		return true
	default:
		return false
	}
}
