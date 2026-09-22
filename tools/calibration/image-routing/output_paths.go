package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// canonicalRepoRoot resolves the original spelling before cleaning and requires
// the checkout root, so Git provenance and filesystem reads use the same tree.
func canonicalRepoRoot(path string) (string, error) {
	root, err := filepath.EvalSymlinks(uncleanedAbsolute(path))
	if err != nil {
		return "", fmt.Errorf("resolve root: %w", err)
	}
	out, err := exec.Command("git", "-C", root, "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", fmt.Errorf("resolve checkout: %w", err)
	}
	top, err := filepath.EvalSymlinks(strings.TrimSpace(string(out)))
	if err != nil {
		return "", fmt.Errorf("resolve checkout root: %w", err)
	}
	if root != top {
		return "", fmt.Errorf("fixture root %q must be the repository top level %q", root, top)
	}
	return root, nil
}

// validateOutputs permits generated files, but never an input, tracked file,
// symlink, or duplicate destination. Return the exact paths to write and exclude.
func validateOutputs(root string, inputs, outputs []string) ([]string, error) {
	resolved := make([]string, 0, len(outputs))
	for _, output := range outputs {
		// Split preserves alias/.. in the parent until symlinks are resolved.
		dir, base := filepath.Split(uncleanedAbsolute(output))
		parent, err := filepath.EvalSymlinks(dir)
		if err != nil {
			return nil, fmt.Errorf("resolve output parent %q: %w", output, err)
		}
		path := filepath.Join(parent, base)
		info, statErr := os.Lstat(path)
		if statErr != nil && !os.IsNotExist(statErr) {
			return nil, fmt.Errorf("inspect output %q: %w", output, statErr)
		}
		if info != nil && !info.Mode().IsRegular() {
			return nil, fmt.Errorf("output %q must be a regular file, not a symlink or directory", output)
		}
		for _, input := range inputs {
			inputPath := filepath.Join(root, filepath.FromSlash(input))
			inputInfo, inputErr := os.Stat(inputPath)
			if inputErr != nil {
				return nil, fmt.Errorf("inspect input %q: %w", input, inputErr)
			}
			if path == inputPath || info != nil && os.SameFile(info, inputInfo) {
				return nil, fmt.Errorf("output %q overlaps input %q", output, input)
			}
		}
		for _, previous := range resolved {
			previousInfo, _ := os.Stat(previous)
			if path == previous || info != nil && previousInfo != nil && os.SameFile(info, previousInfo) {
				return nil, fmt.Errorf("output %q duplicates another report destination", output)
			}
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil, fmt.Errorf("resolve output relative to checkout: %w", err)
		}
		if rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			tracked, trackedErr := trackedFiles(root, []string{filepath.ToSlash(rel)})
			if trackedErr != nil {
				return nil, trackedErr
			}
			// #nosec G204 -- fixed read-only Git command; the path is a literal pathspec after --, not shell code.
			indexed, indexErr := exec.Command("git", "-C", root, "ls-files", "-z", "--", ":(literal)"+filepath.ToSlash(rel)).Output()
			if indexErr != nil {
				return nil, fmt.Errorf("inspect indexed output: %w", indexErr)
			}
			if len(tracked) != 0 || len(indexed) != 0 {
				return nil, fmt.Errorf("output %q is tracked; reports must be generated files", output)
			}
		}
		resolved = append(resolved, path)
	}
	return resolved, nil
}
