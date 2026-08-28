package recipe

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	defaultPageSize        = 50
	maxPageSize            = 200
	maxSearchLength        = 256
	maxMetadataBytes int64 = 256 << 10
	maxConfigBytes   int64 = 16 << 20
	maxProbesBytes   int64 = 8 << 20
	maxDSLBytes      int64 = 8 << 20
	maxREADMEBytes   int64 = 2 << 20
	// Match Router's /api/v1/eval JSON request limit. Stress probes exercise
	// 524K-token bands and can exceed 2 MiB after JSON materialization.
	maxRequestBytes           = 10 << 20
	maxGeneratedTextBytes     = 10 << 20
	maxImageFixtureBytes      = 4 << 20
	maxEvalResponseBytes      = 4 << 20
	defaultCompletionTokens   = 2048
	defaultEvalTimeoutSeconds = 60.0
)

var (
	ErrUnmanaged       = errors.New("active configuration is not a managed recipe")
	ErrProbeNotFound   = errors.New("probe not found")
	ErrInvalid         = errors.New("managed recipe is invalid")
	ErrBadRequest      = errors.New("invalid recipe request")
	ErrStale           = errors.New("managed recipe changed; refresh probes")
	ErrUpstream        = errors.New("router eval request failed")
	ErrActivationState = errors.New("active Recipe activation state is not ready")
)

var recipeFiles = []struct {
	key      string
	name     string
	maxBytes int64
}{
	{key: "metadata", name: "metadata.yaml", maxBytes: maxMetadataBytes},
	{key: "config", name: "config.yaml", maxBytes: maxConfigBytes},
	{key: "probes", name: "probes.yaml", maxBytes: maxProbesBytes},
	{key: "dsl", name: "recipe.dsl", maxBytes: maxDSLBytes},
	{key: "readme", name: "README.md", maxBytes: maxREADMEBytes},
}

type Options struct {
	Directory         string
	Store             *Store
	RuntimeConfigPath string
	RouterAPIURL      string
	HTTPClient        *http.Client
	Evaluator         Evaluator
}

// Service owns the single active Recipe selected when the dashboard process is
// constructed. It only reads the five fixed filenames in Directory and never
// scans or resolves a catalog of sibling directories.
type Service struct {
	directory         string
	store             *Store
	runtimeConfigPath string
	evaluator         Evaluator
	modelResolver     RequestModelResolver
	snapshots         parsedSnapshotCache
}

func NewService(options Options) *Service {
	evaluator := options.Evaluator
	if evaluator == nil {
		client := options.HTTPClient
		if client == nil {
			client = &http.Client{
				CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
					return http.ErrUseLastResponse
				},
			}
		}
		evaluator = &HTTPRouterEvaluator{
			BaseURL:            strings.TrimSpace(options.RouterAPIURL),
			Client:             client,
			CredentialProvider: options.Store,
		}
	}
	service := &Service{
		directory:         filepath.Clean(options.Directory),
		store:             options.Store,
		runtimeConfigPath: filepath.Clean(options.RuntimeConfigPath),
		evaluator:         evaluator,
	}
	if resolver, ok := evaluator.(RequestModelResolver); ok {
		service.modelResolver = resolver
	}
	return service
}

func (s *Service) Describe() (Descriptor, error) {
	snapshot, descriptor, err := s.loadSnapshot()
	if errors.Is(err, ErrUnmanaged) {
		return descriptor, nil
	}
	if errors.Is(err, ErrActivationState) {
		return descriptor, nil
	}
	if err != nil {
		return descriptor, err
	}

	metadata := cloneMetadata(snapshot.metadata)
	descriptor.Metadata = &metadata
	descriptor.README = string(snapshot.files["readme"])
	descriptor.Counts = snapshot.counts
	return descriptor, nil
}

func (s *Service) ListProbes(options ListOptions) (ProbeList, error) {
	snapshot, _, err := s.loadSnapshot()
	if err != nil {
		return ProbeList{}, err
	}
	if err := normalizeListOptions(&options); err != nil {
		return ProbeList{}, err
	}

	filtered := make([]ProbeDetail, 0, len(snapshot.probes))
	for _, probe := range snapshot.probes {
		if matchesFilters(probe, options) {
			filtered = append(filtered, probe)
		}
	}

	total := len(filtered)
	totalPages := 0
	if total > 0 {
		totalPages = (total + options.PageSize - 1) / options.PageSize
	}
	start := total
	pageIndex := options.Page - 1
	// Compare before multiplying so an authenticated caller cannot overflow an
	// int with an arbitrarily large page query and panic the Dashboard process.
	if pageIndex <= total/options.PageSize {
		start = pageIndex * options.PageSize
	}
	end := start + options.PageSize
	if end > total {
		end = total
	}

	items := make([]ProbeSummary, 0, end-start)
	for _, probe := range filtered[start:end] {
		items = append(items, cloneProbeSummary(probe.ProbeSummary))
	}
	return ProbeList{
		Items:        items,
		Page:         options.Page,
		PageSize:     options.PageSize,
		Total:        total,
		TotalPages:   totalPages,
		Facets:       cloneFacets(snapshot.facets),
		RecipeDigest: snapshot.recipeDigest,
	}, nil
}

func (s *Service) Probe(decisionID, variantID string) (ProbeDetail, string, error) {
	snapshot, _, err := s.loadSnapshot()
	if err != nil {
		return ProbeDetail{}, "", err
	}
	probe, ok := snapshot.byID[probeKey(decisionID, variantID)]
	if !ok {
		return ProbeDetail{}, snapshot.recipeDigest, ErrProbeNotFound
	}
	detail := cloneProbeDetail(probe)
	if !detail.Playground.Enabled {
		// Keep exact request components server-side for Validate while exposing
		// only bounded materialization metadata for Eval-only probes.
		detail.Query = ""
		detail.Messages = nil
		detail.Tools = nil
		detail.RawPayloadHidden = true
	}
	return detail, snapshot.recipeDigest, nil
}

func (s *Service) RunPlan(decisionID, variantID string, expectedDigest ...string) (RunPlan, error) {
	snapshot, _, err := s.loadSnapshot()
	if err != nil {
		return RunPlan{}, err
	}
	if digestErr := requireExpectedDigest(snapshot.recipeDigest, expectedDigest); digestErr != nil {
		return RunPlan{}, digestErr
	}
	probe, ok := snapshot.byID[probeKey(decisionID, variantID)]
	if !ok {
		return RunPlan{}, ErrProbeNotFound
	}
	if !probe.Playground.Enabled {
		return RunPlan{}, fmt.Errorf("%w: probe is Eval-only: %s", ErrBadRequest, probe.Playground.Reason)
	}
	request, err := materializeChatRequest(probe)
	if err != nil {
		return RunPlan{}, err
	}
	return RunPlan{
		ProbeID:      probe.ID,
		RecipeDigest: snapshot.recipeDigest,
		Recipe:       normalizeExpectedRecipe(probe.Expected.Recipe),
		Model:        request.Model,
		Messages:     request.Messages,
		Tools:        request.Tools,
		Request:      request,
		Editable:     probe.Editable,
	}, nil
}

func (s *Service) Validate(ctx context.Context, decisionID, variantID string, expectedDigest ...string) (ValidationResult, error) {
	started := time.Now()
	snapshot, _, err := s.loadSnapshot()
	if err != nil {
		return ValidationResult{}, err
	}
	if digestErr := requireExpectedDigest(snapshot.recipeDigest, expectedDigest); digestErr != nil {
		return ValidationResult{}, digestErr
	}
	probe, ok := snapshot.byID[probeKey(decisionID, variantID)]
	if !ok {
		return ValidationResult{}, ErrProbeNotFound
	}
	probe, err = s.resolveActiveProbeModel(ctx, probe)
	if err != nil {
		return ValidationResult{}, err
	}

	result := ValidationResult{
		ProbeID:      probe.ID,
		RecipeDigest: snapshot.recipeDigest,
		Expected:     cloneExpectedAssertions(probe.Expected),
		Actual: ActualOutcome{
			Plugins:           []string{},
			RecommendedModels: []string{},
			MatchedSignals:    map[string][]string{},
			TraceDecisions:    []string{},
		},
		Failures: []string{},
	}
	evalRequest, err := materializeEvalRequest(probe)
	if err != nil {
		return ValidationResult{}, err
	}
	provenanceSession := s.beginValidationProvenance(ctx, snapshot)
	result.Provenance = provenanceSession.result
	evalContext, cancel := context.WithTimeout(ctx, snapshot.requestTimeout)
	defer cancel()
	raw, err := s.evaluator.Evaluate(evalContext, evalRequest)
	result.LatencyMS = time.Since(started).Milliseconds()
	if err != nil {
		result.Provenance = s.finishValidationProvenance(ctx, snapshot, provenanceSession)
		result.Error = err.Error()
		result.Failures = append(result.Failures, "router eval request failed")
		return result, fmt.Errorf("%w: %w", ErrUpstream, err)
	}

	actual, checks, failures, err := compareEvalResponse(raw, probe, snapshot.probes)
	if err != nil {
		result.Provenance = s.finishValidationProvenance(ctx, snapshot, provenanceSession)
		result.Error = err.Error()
		result.Failures = append(result.Failures, "router eval response was invalid")
		return result, fmt.Errorf("%w: %w", ErrUpstream, err)
	}
	result.Actual = actual
	result.Checks = checks
	result.Failures = failures
	result.Passed = len(failures) == 0
	result.Provenance = s.finishValidationProvenance(ctx, snapshot, provenanceSession)
	return result, nil
}

func (s *Service) resolveActiveProbeModel(ctx context.Context, probe ProbeDetail) (ProbeDetail, error) {
	if strings.TrimSpace(probe.Model) != "" || s.modelResolver == nil {
		return probe, nil
	}
	recipeName := normalizeExpectedRecipe(probe.Expected.Recipe)
	model, err := s.modelResolver.ResolveRequestModel(ctx, recipeName)
	if err != nil {
		return ProbeDetail{}, fmt.Errorf("%w: resolve request model for Recipe %q: %w", ErrUpstream, recipeName, err)
	}
	probe.Model = model
	return probe, nil
}

func requireExpectedDigest(actual string, expected []string) error {
	if len(expected) == 0 {
		return nil
	}
	candidate := strings.TrimSpace(expected[0])
	if candidate == "" {
		return fmt.Errorf("%w: expected recipe digest must not be empty", ErrBadRequest)
	}
	if candidate != actual {
		return ErrStale
	}
	return nil
}

type snapshot struct {
	metadata       Metadata
	files          map[string][]byte
	probes         []ProbeDetail
	byID           map[string]ProbeDetail
	facets         Facets
	recipeDigest   string
	counts         Counts
	requestTimeout time.Duration
}

func (s *Service) loadSnapshotLocked() (*snapshot, Descriptor, error) {
	directory, activationState, activeDigest, activationErr := s.resolveDirectory()
	files, health, digests, issues := s.readStableFixedFiles(directory)
	descriptor := Descriptor{
		Managed:            false,
		ActivationState:    activationState,
		ActiveRecipeDigest: activeDigest,
		SourceHealth: SourceHealth{
			Status: "unmanaged",
			Files:  health,
			Issues: []string{},
		},
		Digests: digests,
		Counts:  Counts{},
	}
	cacheKey := completeParsedSnapshotKey(directory, files, health, issues)
	s.snapshots.invalidateUnless(cacheKey)
	if !health["metadata"].Present {
		return unmanagedSnapshotResult(descriptor, issues, activationState, activationErr)
	}
	if err := s.validateSourceRuntimeBindingForDescriptor(activationState, health, files, &descriptor); err != nil {
		return nil, descriptor, err
	}

	descriptor.Managed = true
	issues = appendMissingRecipeFileIssues(health, issues)
	if cached := s.snapshots.lookup(cacheKey); cached != nil {
		descriptor = descriptorForSnapshot(descriptor, cached, digests)
		if activationErr != nil {
			descriptor.SourceHealth.Status = activationState
			descriptor.SourceHealth.Issues = []string{activationIssue(activationState)}
			return nil, descriptor, ErrActivationState
		}
		return cached, descriptor, nil
	}

	metadata, probes, projection, issues := parseRecipeSource(files, issues)

	if len(issues) > 0 {
		descriptor.Metadata = metadataIfPresent(metadata)
		descriptor.SourceHealth.Status = "invalid"
		descriptor.SourceHealth.Issues = stableUnique(issues)
		return nil, descriptor, errors.Join(ErrInvalid, errors.New(strings.Join(descriptor.SourceHealth.Issues, "; ")))
	}

	flattened, byID := flattenProbes(probes)
	if err := resolveModelLessProbeModels(flattened, byID, projection); err != nil {
		descriptor.SourceHealth.Status = "invalid"
		descriptor.SourceHealth.Issues = []string{err.Error()}
		return nil, descriptor, errors.Join(ErrInvalid, err)
	}
	recipeDigest, counts := populateReadyRecipeDescriptor(&descriptor, files, metadata, projection, flattened, digests)
	if activationErr != nil {
		descriptor.SourceHealth.Status = activationState
		descriptor.SourceHealth.Issues = append(descriptor.SourceHealth.Issues, activationIssue(activationState))
		return nil, descriptor, ErrActivationState
	}
	requestTimeoutSeconds := defaultEvalTimeoutSeconds
	if probes.Evaluation.RequestTimeoutSeconds != nil {
		requestTimeoutSeconds = *probes.Evaluation.RequestTimeoutSeconds
	}
	parsed := newParsedRecipeSnapshot(metadata, files, flattened, byID, recipeDigest, counts, requestTimeoutSeconds)
	s.snapshots.store(cacheKey, parsed)
	return parsed, descriptor, nil
}

func populateReadyRecipeDescriptor(
	descriptor *Descriptor,
	files map[string][]byte,
	metadata Metadata,
	projection configProjection,
	probes []ProbeDetail,
	digests Digests,
) (string, Counts) {
	recipeDigest := digestRecipe(files)
	digests.Recipe = recipeDigest
	descriptor.Digests = digests
	descriptor.SourceHealth.Status = "ready"
	descriptor.Metadata = &metadata
	descriptor.README = string(files["readme"])
	counts := projection.counts
	counts.Probes = len(probes)
	descriptor.Counts = counts
	return recipeDigest, counts
}

func newParsedRecipeSnapshot(
	metadata Metadata,
	files map[string][]byte,
	probes []ProbeDetail,
	byID map[string]ProbeDetail,
	recipeDigest string,
	counts Counts,
	requestTimeoutSeconds float64,
) *snapshot {
	return &snapshot{
		metadata:       metadata,
		files:          files,
		probes:         probes,
		byID:           byID,
		facets:         buildFacets(probes),
		recipeDigest:   recipeDigest,
		counts:         counts,
		requestTimeout: time.Duration(requestTimeoutSeconds * float64(time.Second)),
	}
}

func unmanagedSnapshotResult(
	descriptor Descriptor,
	issues []string,
	activationState string,
	activationErr error,
) (*snapshot, Descriptor, error) {
	if len(issues) > 0 {
		descriptor.SourceHealth.Issues = issues
	}
	if activationErr != nil {
		descriptor.SourceHealth.Status = activationState
		descriptor.SourceHealth.Issues = append(descriptor.SourceHealth.Issues, activationIssue(activationState))
		return nil, descriptor, ErrActivationState
	}
	return nil, descriptor, ErrUnmanaged
}

func (s *Service) validateSourceRuntimeBindingForDescriptor(
	activationState string,
	health map[string]FileHealth,
	files map[string][]byte,
	descriptor *Descriptor,
) error {
	if activationState != ActivationNone || !health["config"].Present {
		return nil
	}
	if err := s.validateSourceRuntimeBinding(files["config"]); err != nil {
		descriptor.SourceHealth.Status = "unmanaged"
		descriptor.SourceHealth.Issues = []string{"source Recipe no longer matches the active runtime config"}
		return ErrUnmanaged
	}
	return nil
}

func appendMissingRecipeFileIssues(health map[string]FileHealth, issues []string) []string {
	for _, file := range recipeFiles {
		if !health[file.key].Present {
			issues = append(issues, fmt.Sprintf("required file %s is missing", file.name))
		}
	}
	return issues
}

func parseRecipeSource(
	files map[string][]byte,
	issues []string,
) (Metadata, probeManifest, configProjection, []string) {
	metadata, err := decodeMetadata(files["metadata"])
	if err != nil {
		issues = append(issues, err.Error())
	}
	probes, err := decodeProbes(files["probes"])
	if err != nil {
		issues = append(issues, err.Error())
	}
	projection, err := projectConfig(files["config"])
	if err != nil {
		issues = append(issues, fmt.Sprintf("config.yaml: %v", err))
	}
	if metadata.ID != "" && probes.Name != "" && metadata.ID != probes.Name {
		issues = append(issues, fmt.Sprintf("metadata id %q does not match probes name %q", metadata.ID, probes.Name))
	}
	return metadata, probes, projection, issues
}

func resolveModelLessProbeModels(probes []ProbeDetail, byID map[string]ProbeDetail, projection configProjection) error {
	defaultProbeIndex := 0
	for index := range probes {
		probe := probes[index]
		if probe.Model != "" {
			continue
		}
		expectedRecipe := normalizeExpectedRecipe(probe.Expected.Recipe)
		model := ""
		var err error
		if expectedRecipe == "default" && len(projection.autoModels) > 0 {
			// Only model-less default probes participate in offline-compatible
			// auto-entrypoint round-robin assignment.
			model = projection.autoModels[defaultProbeIndex%len(projection.autoModels)]
			defaultProbeIndex++
		} else {
			// Named recipes use their first executable entrypoint in Dashboard Run.
			model, err = projection.requestModelFor(expectedRecipe)
		}
		if err != nil {
			// A reusable Recipe package can intentionally omit Entrypoints. Its
			// request model is resolved from the active Router when the probe runs.
			continue
		}
		probe.Model = model
		probes[index] = probe
		byID[probeKey(probe.DecisionID, probe.VariantID)] = probe
	}
	return nil
}

func digestBytes(data []byte) string {
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func digestRecipe(files map[string][]byte) string {
	hash := sha256.New()
	for _, file := range recipeFiles {
		_, _ = io.WriteString(hash, file.name)
		_, _ = hash.Write([]byte{0})
		_, _ = hash.Write(files[file.key])
		_, _ = hash.Write([]byte{0})
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func metadataIfPresent(metadata Metadata) *Metadata {
	if metadata.SchemaVersion == "" && metadata.ID == "" && metadata.Name == "" {
		return nil
	}
	return &metadata
}

func normalizeListOptions(options *ListOptions) error {
	if options.Page == 0 {
		options.Page = 1
	}
	if options.Page < 1 {
		return fmt.Errorf("%w: page must be at least 1", ErrBadRequest)
	}
	if options.PageSize == 0 {
		options.PageSize = defaultPageSize
	}
	if options.PageSize < 1 || options.PageSize > maxPageSize {
		return fmt.Errorf("%w: page_size must be between 1 and %d", ErrBadRequest, maxPageSize)
	}
	options.Query = strings.TrimSpace(options.Query)
	if utf8.RuneCountInString(options.Query) > maxSearchLength {
		return fmt.Errorf("%w: q exceeds %d characters", ErrBadRequest, maxSearchLength)
	}
	if err := normalizeListFilters(options); err != nil {
		return err
	}
	options.Shape = strings.ToLower(strings.TrimSpace(options.Shape))
	if options.Shape != "" && options.Shape != "text" && options.Shape != "messages" && options.Shape != "tools" {
		return fmt.Errorf("%w: shape must be text, messages, or tools", ErrBadRequest)
	}
	return nil
}

func normalizeListFilters(options *ListOptions) error {
	options.Recipe = strings.TrimSpace(options.Recipe)
	options.Decision = strings.TrimSpace(options.Decision)
	options.Tag = strings.TrimSpace(options.Tag)
	options.Model = strings.TrimSpace(options.Model)
	for _, filter := range []struct {
		name  string
		value string
	}{
		{name: "recipe", value: options.Recipe},
		{name: "decision", value: options.Decision},
		{name: "tag", value: options.Tag},
		{name: "model", value: options.Model},
	} {
		if utf8.RuneCountInString(filter.value) > maxSearchLength {
			return fmt.Errorf("%w: %s exceeds %d characters", ErrBadRequest, filter.name, maxSearchLength)
		}
	}
	return nil
}

func matchesFilters(probe ProbeDetail, options ListOptions) bool {
	if options.Recipe != "" && normalizeExpectedRecipe(probe.Expected.Recipe) != options.Recipe {
		return false
	}
	if options.Decision != "" && probe.DecisionID != options.Decision {
		return false
	}
	if options.Tag != "" && !contains(probe.Tags, options.Tag) {
		return false
	}
	if options.Model != "" && probe.Model != options.Model {
		return false
	}
	if options.Shape != "" && !contains(probe.RequestShapes, options.Shape) {
		return false
	}
	if options.Query == "" {
		return true
	}
	needle := strings.ToLower(options.Query)
	haystack := strings.ToLower(strings.Join([]string{
		probe.ID,
		probe.DecisionID,
		probe.VariantID,
		probe.Model,
		probe.QueryPreview,
		probe.Notes,
		strings.Join(probe.Tags, " "),
	}, " "))
	return strings.Contains(haystack, needle)
}

func buildFacets(probes []ProbeDetail) Facets {
	facets := Facets{
		Recipes:   map[string]int{},
		Decisions: map[string]int{},
		Tags:      map[string]int{},
		Models:    map[string]int{},
		Shapes:    map[string]int{},
	}
	for _, probe := range probes {
		facets.Recipes[normalizeExpectedRecipe(probe.Expected.Recipe)]++
		facets.Decisions[probe.DecisionID]++
		if probe.Model != "" {
			facets.Models[probe.Model]++
		}
		for _, tag := range probe.Tags {
			facets.Tags[tag]++
		}
		for _, shape := range probe.RequestShapes {
			facets.Shapes[shape]++
		}
	}
	return facets
}
