package handlers

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// openClawSkillPackDir returns the server-owned directory holding known
// skill packs (each pack is a subdirectory containing SKILL.md and optional
// assets). Only these server-controlled copies are used when materializing
// skills into a workspace; content is never extracted from the caller-
// selected image.
func openClawSkillPackDir() string {
	if p := strings.TrimSpace(os.Getenv("OPENCLAW_SKILL_PACKS_PATH")); p != "" {
		return p
	}
	if wd, err := os.Getwd(); err == nil {
		return filepath.Join(wd, "skillpacks")
	}
	return "skillpacks"
}

// validateOpenClawSkillPackID mirrors the strict catalog ID pattern so a pack
// directory name can never carry path separators, traversal segments, or
// aliases.
func validateOpenClawSkillPackID(skillID string) error {
	if !openClawSkillIDPattern.MatchString(skillID) {
		return fmt.Errorf("invalid skill ID %q", skillID)
	}
	return nil
}

// resolveOpenClawSkillPack resolves skillID to a server-owned pack directory
// and proves, after symlink-aware evaluation, that both the pack directory
// and its SKILL.md stay inside the skill-pack root. It rejects unknown IDs,
// aliases, and any symlink that escapes the root.
func resolveOpenClawSkillPack(skillID string) (packDir string, skillFile string, err error) {
	if err := validateOpenClawSkillPackID(skillID); err != nil {
		return "", "", err
	}

	root := openClawSkillPackDir()
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve skill pack root: %w", err)
	}
	rootEval, err := filepath.EvalSymlinks(rootAbs)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve skill pack root: %w", err)
	}

	candidate := filepath.Join(rootEval, skillID)
	candidateEval, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", "", fmt.Errorf("unknown skill ID %q", skillID)
	}
	if !pathWithinDir(candidateEval, rootEval) {
		return "", "", fmt.Errorf("unknown skill ID %q", skillID)
	}

	skillPath := filepath.Join(candidateEval, "SKILL.md")
	skillEval, err := filepath.EvalSymlinks(skillPath)
	if err != nil {
		return "", "", fmt.Errorf("unknown skill ID %q", skillID)
	}
	if !pathWithinDir(skillEval, rootEval) {
		return "", "", fmt.Errorf("unknown skill ID %q", skillID)
	}
	return candidateEval, skillEval, nil
}

// pathWithinDir reports whether target is dir itself or nested inside dir.
// Both paths must already be symlink-resolved and absolute.
func pathWithinDir(target, dir string) bool {
	if target == dir {
		return true
	}
	if dir == "" || dir == "/" {
		return false
	}
	rel, err := filepath.Rel(dir, target)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// copyOpenClawSkillPack copies the server-owned skill pack into the given
// workspace skills directory under the pack's strict ID. Existing files are
// overwritten; destination paths are always derived from the server catalog
// ID and contained inside skillsRoot after symlink-aware resolution.
func copyOpenClawSkillPack(skillID, skillsRoot string) error {
	if err := validateOpenClawSkillPackID(skillID); err != nil {
		return err
	}

	rootAbs, err := filepath.Abs(skillsRoot)
	if err != nil {
		return fmt.Errorf("failed to resolve workspace skills dir: %w", err)
	}
	rootEval, err := filepath.EvalSymlinks(rootAbs)
	if err != nil {
		return fmt.Errorf("failed to resolve workspace skills dir: %w", err)
	}

	packDir, _, err := resolveOpenClawSkillPack(skillID)
	if err != nil {
		return err
	}

	destDir := filepath.Join(rootEval, skillID)
	if !pathWithinDir(destDir, rootEval) {
		return fmt.Errorf("refusing skill destination outside workspace")
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("failed to create skill dir: %w", err)
	}

	return copyDirWithinRoot(packDir, destDir, rootEval)
}

// copyDirWithinRoot recursively copies src into dst, refusing any entry
// (after symlink resolution) that would resolve outside root.
func copyDirWithinRoot(src, dst, root string) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return fmt.Errorf("failed to read skill pack: %w", err)
	}

	// Deterministic order for reproducible provisioning.
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)

	for _, name := range names {
		entryPath := filepath.Join(src, name)
		entryEval, err := filepath.EvalSymlinks(entryPath)
		if err != nil {
			return fmt.Errorf("failed to resolve %s: %w", name, err)
		}
		if !pathWithinDir(entryEval, root) && !pathWithinDir(entryEval, src) {
			// Symlinked pack content may legitimately point elsewhere inside
			// the pack, but never outside the server-owned root.
			return fmt.Errorf("skill pack entry %q resolves outside the skill pack root", name)
		}

		info, err := os.Lstat(entryPath)
		if err != nil {
			return fmt.Errorf("failed to stat %s: %w", name, err)
		}

		targetPath := filepath.Join(dst, name)
		switch {
		case info.IsDir():
			if err := os.MkdirAll(targetPath, 0o755); err != nil {
				return fmt.Errorf("failed to create dir %s: %w", name, err)
			}
			if err := copyDirWithinRoot(entryEval, targetPath, root); err != nil {
				return err
			}
		case info.Mode().IsRegular():
			data, err := os.ReadFile(entryEval)
			if err != nil {
				return fmt.Errorf("failed to read %s: %w", name, err)
			}
			if err := os.WriteFile(targetPath, data, 0o644); err != nil {
				return fmt.Errorf("failed to write %s: %w", name, err)
			}
		default:
			// Skip sockets, devices, and dangling symlinks: never copy them
			// into a provisioned workspace.
		}
	}
	return nil
}
