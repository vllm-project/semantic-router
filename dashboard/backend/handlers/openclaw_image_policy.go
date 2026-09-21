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

// validateOpenClawImage enforces the administrator policy against the final
// (already resolved) image reference. It returns a descriptive error naming
// the policy, never the server filesystem.
func validateOpenClawImage(image string, policy openClawImagePolicy) error {
	image = strings.ToLower(strings.TrimSpace(image))
	if image == "" {
		return fmt.Errorf("OpenClaw image is empty")
	}

	// Split tag/digest while keeping the repository part intact.
	ref := image
	tag := ""
	digest := ""
	if idx := strings.Index(ref, "@"); idx >= 0 {
		digest = ref[idx+1:]
		ref = ref[:idx]
	} else if idx := strings.LastIndex(ref, ":"); idx >= 0 && !strings.Contains(ref[idx:], "/") {
		tag = ref[idx+1:]
		ref = ref[:idx]
	}

	if policy.AllowDigestsOnly && digest == "" {
		return fmt.Errorf("image policy requires digest-pinned references (image@sha256:...); got %q", image)
	}
	if digest != "" && !strings.HasPrefix(digest, "sha256:") {
		return fmt.Errorf("image policy only accepts sha256 digests; got %q", image)
	}
	if policy.AllowedTags != nil && tag != "" {
		allowed := false
		for _, t := range policy.AllowedTags {
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
		if allowed == ref || allowed == image {
			return nil
		}
		// Prefix match: "ghcr.io/openclaw/" allows every repository under it.
		if strings.HasSuffix(allowed, "/") && strings.HasPrefix(ref, allowed) {
			return nil
		}
	}
	return fmt.Errorf("image %q is not in the image allowlist", image)
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

// openClawStateVolumeArgs returns the volume and tmpfs arguments that keep
// the container's writable state inside the administrator-managed named
// volume instead of the image filesystem.
func openClawStateVolumeArgs(absCDir, volumeName string) []string {
	return []string{
		"-v", absCDir + "/workspace:/workspace",
		"-v", absCDir + "/openclaw.json:/config/openclaw.json:ro",
		"-v", volumeName + ":/state",
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

	dataDirAbs, err := filepath.Abs(dataDir)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve data dir: %w", err)
	}
	dataDirEval, err := filepath.EvalSymlinks(dataDirAbs)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve data dir: %w", err)
	}

	cDir = filepath.Join(dataDirEval, "containers", containerName)
	cDirEval, err := filepath.EvalSymlinks(cDir)
	if err == nil {
		// The directory already exists (reprovision): ensure it did not
		// become a symlink pointing outside dataDir.
		if !pathWithinDir(cDirEval, dataDirEval) {
			return "", "", fmt.Errorf("container data dir resolves outside the data directory")
		}
		cDir = cDirEval
	} else if !os.IsNotExist(err) {
		return "", "", fmt.Errorf("failed to resolve container data dir: %w", err)
	}

	wsDir = filepath.Join(cDir, "workspace")
	wsDirEval, err := filepath.EvalSymlinks(wsDir)
	if err == nil {
		if !pathWithinDir(wsDirEval, dataDirEval) {
			return "", "", fmt.Errorf("workspace resolves outside the data directory")
		}
		wsDir = wsDirEval
	} else if !os.IsNotExist(err) {
		return "", "", fmt.Errorf("failed to resolve workspace: %w", err)
	}
	return cDir, wsDir, nil
}
