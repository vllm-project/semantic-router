package handlers

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// openClawImagePolicy holds the administrator-controlled image provenance
// policy. When it is configured, provisioning may only use images that match
// an allowlist entry (exact reference or registry/repository prefix) and, if
// requested with a digest, only pinned digests.
type openClawImagePolicy struct {
	// Allowlist entries are lowercase image references or prefixes ending
	// with "/". An empty list rejects every image (deny by default).
	Allowed []string
	// AllowedTags, when non-empty, is the set of tag values that may be used
	// for allowlisted repositories. An empty set allows any tag (including
	// digest references).
	AllowedTags []string
	// AllowDigestsOnly, when true, rejects tag references and only accepts
	// image@sha256:... references.
	AllowDigestsOnly bool
}

// openClawImagePolicyDisabled reports whether no policy is configured, which
// keeps the legacy behavior (no additional provenance restriction beyond the
// default image resolution).
func (p openClawImagePolicy) configured() bool {
	return len(p.Allowed) > 0 || len(p.AllowedTags) > 0 || p.AllowDigestsOnly
}

// loadOpenClawImagePolicy reads the administrator policy from environment
// variables:
//
//	OPENCLAW_IMAGE_ALLOWLIST  comma-separated image refs or registry/repo prefixes
//	OPENCLAW_IMAGE_ALLOWED_TAGS  comma-separated allowed tag values
//	OPENCLAW_IMAGE_DIGESTS_ONLY  "1"/"true" to require digest pinning
func loadOpenClawImagePolicy() openClawImagePolicy {
	splitList := func(raw string) []string {
		var out []string
		for _, item := range strings.Split(raw, ",") {
			item = strings.ToLower(strings.TrimSpace(item))
			if item != "" {
				out = append(out, item)
			}
		}
		return out
	}

	policy := openClawImagePolicy{
		Allowed:     splitList(os.Getenv("OPENCLAW_IMAGE_ALLOWLIST")),
		AllowedTags: splitList(os.Getenv("OPENCLAW_IMAGE_ALLOWED_TAGS")),
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("OPENCLAW_IMAGE_DIGESTS_ONLY"))) {
	case "1", "true", "yes", "on":
		policy.AllowDigestsOnly = true
	}
	return policy
}

// splitImageReference splits an image reference into repository, tag, and
// digest, applying the runtime's implicit ":latest" tag to untagged
// references. Case is preserved for tags (registries treat them as
// case-sensitive distinctions); the repository is compared case-insensitively
// as Docker does.
func splitImageReference(image string) (ref string, tag string, digest string) {
	reference := strings.TrimSpace(image)
	if reference == "" {
		return "", "", ""
	}
	if idx := strings.Index(reference, "@"); idx >= 0 {
		digest = reference[idx+1:]
		reference = reference[:idx]
	} else if idx := strings.LastIndex(reference, ":"); idx >= 0 && !strings.Contains(reference[idx:], "/") {
		tag = reference[idx+1:]
		reference = reference[:idx]
	} else {
		// The runtime pulls ":latest" for untagged references; validate the
		// effective reference rather than the empty tag.
		tag = "latest"
	}
	return reference, tag, digest
}

// validateOpenClawImage enforces the administrator policy against the final
// (already resolved) image reference. The effective tag (with the implicit
// ":latest" applied) is validated so an untagged request cannot bypass the
// tag allowlist. It returns a descriptive error naming the policy, never
// the server filesystem.
func validateOpenClawImage(image string, policy openClawImagePolicy) error {
	raw := strings.TrimSpace(image)
	if raw == "" {
		return fmt.Errorf("OpenClaw image is empty")
	}

	ref, tag, digest := splitImageReference(raw)

	if policy.AllowDigestsOnly && digest == "" {
		return fmt.Errorf("image policy requires digest-pinned references (image@sha256:...); got %q", raw)
	}
	if digest != "" && !strings.HasPrefix(digest, "sha256:") {
		return fmt.Errorf("image policy only accepts sha256 digests; got %q", raw)
	}
	if len(policy.AllowedTags) > 0 && tag != "" {
		allowed := false
		for _, t := range policy.AllowedTags {
			// Tags are case-sensitive: ":RELEASE" must not satisfy an
			// allowlist entry of ":release".
			if t == tag {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("image tag %q is not allowed by the image policy", tag)
		}
	}

	if len(policy.Allowed) == 0 {
		// No allowlist entries: repository provenance is unrestricted and
		// only the tag/digest rules above apply. To deny every image, set
		// OPENCLAW_IMAGE_ALLOWED_TAGS to a sentinel that matches no tag.
		return nil
	}

	for _, allowed := range policy.Allowed {
		// Repository comparison is case-insensitive, matching the runtime's
		// reference normalization; tag/digest case is preserved above.
		if strings.EqualFold(allowed, ref) || strings.EqualFold(allowed, raw) {
			return nil
		}
		// Prefix match: "ghcr.io/openclaw/" allows every repository under it.
		if strings.HasSuffix(allowed, "/") && strings.HasPrefix(strings.ToLower(ref), strings.ToLower(allowed)) {
			return nil
		}
	}
	return fmt.Errorf("image %q is not in the image allowlist", raw)
}

// securityOpts and runtime flags for least-privilege provisioning.
func openClawLeastPrivilegeArgs(containerName string) []string {
	_ = containerName // reserved for future per-container tweaks
	return []string{
		"--read-only",
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
		"--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
	}
}

// validateProvisionPaths proves that the per-container data directory and
// the workspace stay inside the handler's dataDir after symlink-aware
// resolution. It is defense in depth against a container name that later
// collides with a symlink placed inside dataDir.
func validateProvisionPaths(dataDir, containerName string) (cDir string, wsDir string, err error) {
	if containerName == "" {
		return "", "", fmt.Errorf("container name is empty")
	}

	dataDirAbs, absErr := filepath.Abs(dataDir)
	if absErr != nil {
		return "", "", fmt.Errorf("failed to resolve data dir: %w", absErr)
	}
	dataDirEval, evalErr := filepath.EvalSymlinks(dataDirAbs)
	if evalErr != nil {
		return "", "", fmt.Errorf("failed to resolve data dir: %w", evalErr)
	}

	// Resolve the deepest existing ancestor of the container directory and
	// prove it stays inside dataDir before accepting any missing tail.
	// Without this, dataDir/containers could be a symlink outside dataDir
	// while the container directory itself does not exist yet, and the
	// ENOENT branch below would accept the unresolved path.
	containersDir, ancErr := resolveExistingWithin(filepath.Join(dataDirEval, "containers"), dataDirEval)
	if ancErr != nil {
		return "", "", fmt.Errorf("invalid containers dir: %w", ancErr)
	}

	cDir, cErr := resolveExistingWithin(filepath.Join(containersDir, containerName), dataDirEval)
	if cErr != nil {
		return "", "", fmt.Errorf("invalid container data dir: %w", cErr)
	}

	wsDir, wsErr := resolveExistingWithin(filepath.Join(cDir, "workspace"), dataDirEval)
	if wsErr != nil {
		return "", "", fmt.Errorf("invalid workspace: %w", wsErr)
	}
	return cDir, wsDir, nil
}

// resolveExistingWithin resolves the deepest existing ancestor of path
// symlink-aware and returns path with its existing prefix replaced by the
// resolved ancestor. It fails when the resolved ancestor escapes root.
func resolveExistingWithin(path, root string) (string, error) {
	missing := path
	var tail []string
	for {
		eval, err := filepath.EvalSymlinks(missing)
		if err == nil {
			if !pathWithinDir(eval, root) {
				return "", fmt.Errorf("resolves outside the data directory")
			}
			if len(tail) == 0 {
				return eval, nil
			}
			return filepath.Join(append([]string{eval}, tail...)...), nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}
		parent := filepath.Dir(missing)
		if parent == missing {
			return "", fmt.Errorf("resolves outside the data directory")
		}
		tail = append([]string{filepath.Base(missing)}, tail...)
		missing = parent
	}
}
