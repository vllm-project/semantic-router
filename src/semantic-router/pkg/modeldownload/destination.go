package modeldownload

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const modelDownloadRoot = "models"

// validateModelDownloadLocalPath keeps configuration-owned download
// destinations inside the router's relative models/ inventory. Local model
// paths outside that inventory remain valid runtime inputs, but they are
// intentionally never download targets.
func validateModelDownloadLocalPath(localPath string) error {
	if localPath == "" {
		return fmt.Errorf("model download path is empty")
	}
	if strings.IndexFunc(localPath, unicode.IsControl) >= 0 {
		return fmt.Errorf("model download path contains control characters")
	}
	if filepath.IsAbs(localPath) {
		return fmt.Errorf("model download path %q must be relative", localPath)
	}
	cleaned := filepath.Clean(localPath)
	if cleaned != localPath {
		return fmt.Errorf("model download path %q must be canonical", localPath)
	}
	relative, err := filepath.Rel(modelDownloadRoot, cleaned)
	if err != nil || relative == "." || relative == "" || pathEscapesRoot(relative) {
		return fmt.Errorf("model download path %q must stay below %s/", localPath, modelDownloadRoot)
	}
	return nil
}

// validateModelDownloadDestination also resolves every currently existing
// symlink component. This prevents a syntactically safe models/foo target from
// escaping through models/foo (or one of its parents) when handed to the
// external Hugging Face downloader.
func validateModelDownloadDestination(localPath string) error {
	if err := validateModelDownloadLocalPath(localPath); err != nil {
		return err
	}

	root, err := canonicalModelDownloadRoot()
	if err != nil {
		return err
	}
	target := filepath.Join(filepath.Dir(root), localPath)
	resolvedTarget, err := resolveThroughExistingAncestor(target)
	if err != nil {
		return fmt.Errorf("resolve model download destination: %w", err)
	}
	if !isStrictPathDescendant(root, resolvedTarget) {
		return fmt.Errorf("model download path %q resolves outside %s/", localPath, modelDownloadRoot)
	}
	return nil
}

func canonicalModelDownloadRoot() (string, error) {
	workingDirectory, err := filepath.Abs(".")
	if err != nil {
		return "", fmt.Errorf("resolve working directory: %w", err)
	}
	canonicalWorkingDirectory, err := filepath.EvalSymlinks(workingDirectory)
	if err != nil {
		return "", fmt.Errorf("resolve working directory: %w", err)
	}

	// The inventory root is anchored to the canonical working directory. Do
	// not accept a models symlink and then redefine its target as the trusted
	// root: doing so would make an entire-tree escape look contained.
	root := filepath.Join(canonicalWorkingDirectory, modelDownloadRoot)
	if err := validateModelDownloadRoot(root); err != nil {
		return "", err
	}
	return root, nil
}

func validateModelDownloadRoot(root string) error {
	rootInfo, err := os.Lstat(root)
	if err == nil {
		if rootInfo.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("model download root %s must not be a symlink", modelDownloadRoot)
		}
		if !rootInfo.IsDir() {
			return fmt.Errorf("model download root %s must be a directory", modelDownloadRoot)
		}
		return nil
	}
	if os.IsNotExist(err) {
		return nil
	}
	return fmt.Errorf("inspect model download root: %w", err)
}

func isStrictPathDescendant(root, target string) bool {
	relative, err := filepath.Rel(root, target)
	if err != nil || relative == "." || relative == "" {
		return false
	}
	return !pathEscapesRoot(relative)
}

func pathEscapesRoot(relative string) bool {
	return relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative)
}

// resolveThroughExistingAncestor resolves symlinks even when the final target
// does not exist yet. filepath.EvalSymlinks alone cannot do that, which is the
// normal state immediately before a first model download.
func resolveThroughExistingAncestor(path string) (string, error) {
	path = filepath.Clean(path)
	missing := make([]string, 0, 4)
	for {
		_, err := os.Lstat(path)
		switch {
		case err == nil:
			resolved, resolveErr := filepath.EvalSymlinks(path)
			if resolveErr != nil {
				return "", resolveErr
			}
			for i := len(missing) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, missing[i])
			}
			return filepath.Clean(resolved), nil
		case os.IsNotExist(err):
			parent := filepath.Dir(path)
			if parent == path {
				return "", err
			}
			missing = append(missing, filepath.Base(path))
			path = parent
		default:
			return "", err
		}
	}
}
