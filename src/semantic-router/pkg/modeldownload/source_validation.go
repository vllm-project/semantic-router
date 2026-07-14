package modeldownload

import (
	"fmt"
	"strings"
)

const (
	maxHFRepoIDLength   = 96
	maxHFRevisionLength = 200
)

// validateModelSource admits only canonical Hugging Face identifiers before
// they cross the exec/logging seam. exec.Command already avoids a shell, but a
// positional repo ID beginning with '-' can still be interpreted as a CLI
// option, while control characters can forge log lines.
func validateModelSource(spec ModelSpec) error {
	if err := validateHFRepoID(spec.RepoID); err != nil {
		return err
	}
	if err := validateHFRevision(spec.Revision); err != nil {
		return err
	}
	return nil
}

func validateHFRepoID(repoID string) error {
	if repoID == "" {
		return fmt.Errorf("hugging Face repo ID is empty")
	}
	if len(repoID) > maxHFRepoIDLength {
		return fmt.Errorf("hugging Face repo ID exceeds %d bytes", maxHFRepoIDLength)
	}
	if strings.TrimSpace(repoID) != repoID {
		return fmt.Errorf("hugging Face repo ID %q contains surrounding whitespace", repoID)
	}

	segments := strings.Split(repoID, "/")
	if len(segments) < 1 || len(segments) > 2 {
		return fmt.Errorf("hugging Face repo ID %q must have one or two segments", repoID)
	}
	for _, segment := range segments {
		if err := validateHFNameSegment(segment); err != nil {
			return fmt.Errorf("invalid Hugging Face repo ID %q: %w", repoID, err)
		}
	}
	if strings.HasSuffix(strings.ToLower(repoID), ".git") {
		return fmt.Errorf("hugging Face repo ID %q must not end in .git", repoID)
	}
	return nil
}

func validateHFNameSegment(segment string) error {
	if segment == "" {
		return fmt.Errorf("empty path segment")
	}
	if segment[0] == '.' || segment[0] == '-' || segment[len(segment)-1] == '.' || segment[len(segment)-1] == '-' {
		return fmt.Errorf("segments must not start or end with '.' or '-'")
	}
	if strings.Contains(segment, "..") || strings.Contains(segment, "--") {
		return fmt.Errorf("segments must not contain '..' or '--'")
	}
	for _, char := range []byte(segment) {
		if !isHFNameByte(char) {
			return fmt.Errorf("segments may contain only ASCII letters, digits, '_', '-', and '.'")
		}
	}
	return nil
}

func validateHFRevision(revision string) error {
	if revision == "" {
		return nil
	}
	if err := validateHFRevisionEnvelope(revision); err != nil {
		return err
	}
	if !hasOnlyHFRevisionBytes(revision) {
		return fmt.Errorf("hugging Face revision %q contains unsupported characters", revision)
	}
	for _, segment := range strings.Split(revision, "/") {
		if !isCanonicalHFRevisionSegment(segment) {
			return fmt.Errorf("hugging Face revision %q contains a non-canonical segment", revision)
		}
	}
	return nil
}

func validateHFRevisionEnvelope(revision string) error {
	if len(revision) > maxHFRevisionLength {
		return fmt.Errorf("hugging Face revision exceeds %d bytes", maxHFRevisionLength)
	}
	if strings.TrimSpace(revision) != revision {
		return fmt.Errorf("hugging Face revision %q contains surrounding whitespace", revision)
	}
	if !isHFRevisionBoundaryByte(revision[0]) || !isHFRevisionBoundaryByte(revision[len(revision)-1]) {
		return fmt.Errorf("hugging Face revision %q has a non-canonical boundary", revision)
	}
	if strings.Contains(revision, "..") || strings.Contains(revision, "//") || strings.Contains(revision, "@{") {
		return fmt.Errorf("hugging Face revision %q is not canonical", revision)
	}
	return nil
}

func hasOnlyHFRevisionBytes(revision string) bool {
	for _, char := range []byte(revision) {
		if !isHFRevisionByte(char) {
			return false
		}
	}
	return true
}

func isCanonicalHFRevisionSegment(segment string) bool {
	return segment != "" &&
		!strings.HasPrefix(segment, ".") && !strings.HasSuffix(segment, ".") &&
		!strings.HasPrefix(segment, "-") && !strings.HasSuffix(segment, "-") &&
		!strings.HasSuffix(strings.ToLower(segment), ".lock")
}

func isHFNameByte(char byte) bool {
	return char >= 'a' && char <= 'z' ||
		char >= 'A' && char <= 'Z' ||
		char >= '0' && char <= '9' ||
		char == '_' || char == '-' || char == '.'
}

func isHFRevisionByte(char byte) bool {
	return isHFNameByte(char) || char == '/'
}

func isHFRevisionBoundaryByte(char byte) bool {
	return char >= 'a' && char <= 'z' ||
		char >= 'A' && char <= 'Z' ||
		char >= '0' && char <= '9' ||
		char == '_'
}
