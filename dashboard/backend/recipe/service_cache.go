package recipe

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
)

const stableSnapshotReadAttempts = 3

// parsedSnapshotCache owns one immutable parsed source snapshot for the
// process-scoped Recipe service. The mutex covers both fingerprint collection
// and cache replacement so concurrent requests cannot publish an older source
// snapshot after a newer one.
type parsedSnapshotCache struct {
	mu    sync.Mutex
	key   string
	value *snapshot
}

func (s *Service) loadSnapshot() (*snapshot, Descriptor, error) {
	s.snapshots.mu.Lock()
	defer s.snapshots.mu.Unlock()
	return s.loadSnapshotLocked()
}

// lookup must be called while snapshots.mu is held. A different or invalid
// exact-five key invalidates every cached projection in one assignment.
func (c *parsedSnapshotCache) lookup(key string) *snapshot {
	c.invalidateUnless(key)
	if c.value == nil {
		return nil
	}
	return c.value
}

// invalidateUnless must be called while snapshots.mu is held.
func (c *parsedSnapshotCache) invalidateUnless(key string) {
	if key != "" && c.key == key {
		return
	}
	c.key = ""
	c.value = nil
}

// store must be called while snapshots.mu is held. snapshot is constructed
// completely before publication and is read-only after this point.
func (c *parsedSnapshotCache) store(key string, value *snapshot) {
	if key == "" || value == nil {
		c.key = ""
		c.value = nil
		return
	}
	c.key = key
	c.value = value
}

// readStableFixedFiles verifies the content-derived state of all five fixed
// files twice before parsing or reusing a snapshot. A source being replaced
// continuously fails closed instead of constructing a mixed-generation view.
func (s *Service) readStableFixedFiles(directory string) (map[string][]byte, map[string]FileHealth, Digests, []string) {
	files, health, digests, issues := s.readFixedFiles(directory)
	fingerprint := fixedFileReadFingerprint(health, issues)
	for attempt := 1; attempt < stableSnapshotReadAttempts; attempt++ {
		nextFiles, nextHealth, nextDigests, nextIssues := s.readFixedFiles(directory)
		nextFingerprint := fixedFileReadFingerprint(nextHealth, nextIssues)
		if nextFingerprint == fingerprint {
			return nextFiles, nextHealth, nextDigests, nextIssues
		}
		files, health, digests, issues = nextFiles, nextHealth, nextDigests, nextIssues
		fingerprint = nextFingerprint
	}
	issues = append(issues, "managed Recipe source changed while its fixed files were being read")
	return files, health, digests, stableUnique(issues)
}

func fixedFileReadFingerprint(health map[string]FileHealth, issues []string) string {
	hash := sha256.New()
	for _, file := range recipeFiles {
		state := health[file.key]
		_, _ = io.WriteString(hash, file.name)
		_, _ = hash.Write([]byte{0})
		_, _ = io.WriteString(hash, strconv.FormatBool(state.Present))
		_, _ = hash.Write([]byte{0})
		_, _ = io.WriteString(hash, state.Digest)
		_, _ = hash.Write([]byte{0})
		_, _ = io.WriteString(hash, strconv.FormatInt(state.SizeBytes, 10))
		_, _ = hash.Write([]byte{0})
	}
	stableIssues := append([]string(nil), issues...)
	sort.Strings(stableIssues)
	for _, issue := range stableIssues {
		_, _ = io.WriteString(hash, issue)
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func parsedSnapshotKey(directory string, files map[string][]byte) string {
	return filepath.Clean(directory) + "\x00" + digestRecipe(files)
}

func completeParsedSnapshotKey(
	directory string,
	files map[string][]byte,
	health map[string]FileHealth,
	issues []string,
) string {
	if len(issues) != 0 {
		return ""
	}
	for _, file := range recipeFiles {
		if !health[file.key].Present {
			return ""
		}
	}
	return parsedSnapshotKey(directory, files)
}

func descriptorForSnapshot(descriptor Descriptor, cached *snapshot, digests Digests) Descriptor {
	digests.Recipe = cached.recipeDigest
	metadata := cloneMetadata(cached.metadata)
	descriptor.Managed = true
	descriptor.Metadata = &metadata
	descriptor.README = string(cached.files["readme"])
	descriptor.Digests = digests
	descriptor.Counts = cached.counts
	descriptor.SourceHealth.Status = "ready"
	descriptor.SourceHealth.Issues = []string{}
	return descriptor
}

func cloneMetadata(source Metadata) Metadata {
	cloned := source
	cloned.Authors = append([]Author(nil), source.Authors...)
	for index := range cloned.Authors {
		cloned.Authors[index].Email = cloneStringPointer(source.Authors[index].Email)
		cloned.Authors[index].URL = cloneStringPointer(source.Authors[index].URL)
	}
	cloned.Tags = append([]string(nil), source.Tags...)
	cloned.Links.Homepage = cloneStringPointer(source.Links.Homepage)
	cloned.Links.Documentation = cloneStringPointer(source.Links.Documentation)
	return cloned
}

func cloneStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneProbeSummary(source ProbeSummary) ProbeSummary {
	cloned := source
	cloned.RequestShapes = append([]string(nil), source.RequestShapes...)
	cloned.Tags = append([]string(nil), source.Tags...)
	cloned.Expected = cloneExpectedAssertions(source.Expected)
	return cloned
}

func cloneProbeDetail(source ProbeDetail) ProbeDetail {
	cloned := source
	cloned.ProbeSummary = cloneProbeSummary(source.ProbeSummary)
	cloned.Messages = cloneObjects(source.Messages)
	cloned.Tools = cloneObjects(source.Tools)
	if source.Padding != nil {
		padding := *source.Padding
		cloned.Padding = &padding
	}
	if source.GeneratedText != nil {
		generatedText := *source.GeneratedText
		cloned.GeneratedText = &generatedText
	}
	if source.ImageFixtures != nil {
		cloned.ImageFixtures = make(map[string]ImageFixtureMetadata, len(source.ImageFixtures))
		for name, fixture := range source.ImageFixtures {
			cloned.ImageFixtures[name] = fixture
		}
	}
	if source.imageFixtures != nil {
		cloned.imageFixtures = make(map[string]probeImageFixture, len(source.imageFixtures))
		for name, fixture := range source.imageFixtures {
			cloned.imageFixtures[name] = fixture
		}
	}
	return cloned
}

func cloneExpectedAssertions(source ExpectedAssertions) ExpectedAssertions {
	cloned := source
	cloned.Plugins = append([]string(nil), source.Plugins...)
	cloned.ForbiddenPlugins = append([]string(nil), source.ForbiddenPlugins...)
	cloned.Signals = cloneSignalMap(source.Signals)
	cloned.ForbiddenSignals = cloneSignalMap(source.ForbiddenSignals)
	return cloned
}

func cloneSignalMap(source map[string][]string) map[string][]string {
	if source == nil {
		return nil
	}
	cloned := make(map[string][]string, len(source))
	for key, values := range source {
		cloned[key] = append([]string(nil), values...)
	}
	return cloned
}

func cloneFacets(source Facets) Facets {
	return Facets{
		Recipes:   cloneCountMap(source.Recipes),
		Decisions: cloneCountMap(source.Decisions),
		Tags:      cloneCountMap(source.Tags),
		Models:    cloneCountMap(source.Models),
		Shapes:    cloneCountMap(source.Shapes),
	}
}

func cloneCountMap(source map[string]int) map[string]int {
	cloned := make(map[string]int, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}

func cloneJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		cloned := make(map[string]any, len(typed))
		for key, child := range typed {
			cloned[key] = cloneJSONValue(child)
		}
		return cloned
	case []any:
		cloned := make([]any, len(typed))
		for index, child := range typed {
			cloned[index] = cloneJSONValue(child)
		}
		return cloned
	case []map[string]any:
		return cloneObjects(typed)
	default:
		return typed
	}
}
