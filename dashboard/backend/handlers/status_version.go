package handlers

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
)

var (
	dashboardStatusVersion   string
	readStatusBuildInfo      = debug.ReadBuildInfo
	statusVersionSearchRoots = defaultStatusVersionSearchRoots
	pyprojectVersionPattern  = regexp.MustCompile(`(?m)^version\s*=\s*"([^"]+)"`)
)

func statusVersion() string {
	if version := firstNonEmptyVersion(
		dashboardStatusVersion,
		os.Getenv("VLLM_SR_VERSION"),
		os.Getenv("DASHBOARD_VERSION"),
	); version != "" {
		return normalizeStatusVersion(version)
	}

	if version, ok := readProjectVersion(); ok {
		return normalizeStatusVersion(developmentStatusVersion(version))
	}

	if version := buildInfoMainVersion(); version != "" {
		return normalizeStatusVersion(version)
	}

	return "unknown"
}

func firstNonEmptyVersion(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func normalizeStatusVersion(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return "unknown"
	}
	if strings.HasPrefix(version, "v") {
		return version
	}
	return "v" + version
}

func developmentStatusVersion(version string) string {
	version = strings.TrimPrefix(strings.TrimSpace(version), "v")
	if version == "" || strings.Contains(version, "-") {
		return version
	}

	suffix := "dev.local"
	if revision := buildInfoSetting("vcs.revision"); revision != "" {
		if len(revision) > 7 {
			revision = revision[:7]
		}
		suffix = "dev." + revision
		if buildInfoSetting("vcs.modified") == "true" {
			suffix += ".dirty"
		}
	}

	return version + "-" + suffix
}

func readProjectVersion() (string, bool) {
	for _, root := range statusVersionSearchRoots() {
		for _, candidate := range ancestorPyprojectCandidates(root) {
			content, err := os.ReadFile(candidate)
			if err != nil {
				continue
			}
			match := pyprojectVersionPattern.FindStringSubmatch(string(content))
			if len(match) == 2 {
				return match[1], true
			}
		}
	}
	return "", false
}

func defaultStatusVersionSearchRoots() []string {
	roots := make([]string, 0, 2)
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, cwd)
	}
	if _, filename, _, ok := runtime.Caller(0); ok {
		roots = append(roots, filepath.Dir(filename))
	}
	return roots
}

func ancestorPyprojectCandidates(root string) []string {
	root = filepath.Clean(root)
	candidates := make([]string, 0, 8)
	seen := map[string]struct{}{}

	for {
		candidate := filepath.Join(root, "src", "vllm-sr", "pyproject.toml")
		if _, ok := seen[candidate]; !ok {
			candidates = append(candidates, candidate)
			seen[candidate] = struct{}{}
		}

		parent := filepath.Dir(root)
		if parent == root {
			break
		}
		root = parent
	}

	return candidates
}

func buildInfoMainVersion() string {
	info, ok := readStatusBuildInfo()
	if !ok || info == nil {
		return ""
	}
	version := strings.TrimSpace(info.Main.Version)
	if version == "" || version == "(devel)" {
		return ""
	}
	return version
}

func buildInfoSetting(key string) string {
	info, ok := readStatusBuildInfo()
	if !ok || info == nil {
		return ""
	}
	for _, setting := range info.Settings {
		if setting.Key == key {
			return strings.TrimSpace(setting.Value)
		}
	}
	return ""
}
