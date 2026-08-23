package routingmanagement

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	BuiltInRecipeDistributionRelativeDirectory = "recipes/built-in/latest/mom-v1"
	builtInRecipeConfigFile                    = "config.yaml"
	builtInRecipeMetadataFile                  = "metadata.yaml"
	defaultBuiltInRecipeReconcileInterval      = 15 * time.Second
	maximumBuiltInRecipeAssetBytes             = 8 << 20
	maximumBuiltInRecipeNamespaceBatch         = 256
)

// BuiltInRecipeDistribution is the validated, content-addressed value loaded
// from the Router image. It contains only reusable Recipe policy. Physical
// Models and Entrypoints remain namespace-owned Management resources.
type BuiltInRecipeDistribution struct {
	ID          string
	Name        string
	Version     string
	AssetDigest string
	Recipes     []BuiltInRecipe
}

type BuiltInRecipe struct {
	SourceID       string
	SourceRevision int64
	Input          RecipeInput
	RecipeDigest   string
}

type builtInRecipeMetadata struct {
	SchemaVersion string                `yaml:"schema_version"`
	ID            string                `yaml:"id"`
	Name          string                `yaml:"name"`
	Version       string                `yaml:"version"`
	Description   string                `yaml:"description"`
	Authors       []builtInRecipeAuthor `yaml:"authors"`
	License       string                `yaml:"license"`
	Tags          []string              `yaml:"tags"`
	Links         map[string]string     `yaml:"links"`
}

type builtInRecipeAuthor struct {
	Name string `yaml:"name"`
}

// LoadBuiltInRecipeDistribution reads the one canonical distribution directory
// shipped in Router images. Metadata and config are hashed together so a
// released version cannot be changed in place without a startup conflict.
func LoadBuiltInRecipeDistribution(directory string) (BuiltInRecipeDistribution, error) {
	directory = filepath.Clean(strings.TrimSpace(directory))
	if directory == "." || directory == "" {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: built-in Recipe distribution directory is required", ErrInvalid)
	}
	metadataBytes, err := readBoundedDistributionFile(filepath.Join(directory, builtInRecipeMetadataFile))
	if err != nil {
		return BuiltInRecipeDistribution{}, err
	}
	configBytes, err := readBoundedDistributionFile(filepath.Join(directory, builtInRecipeConfigFile))
	if err != nil {
		return BuiltInRecipeDistribution{}, err
	}
	return ParseBuiltInRecipeDistribution(metadataBytes, configBytes)
}

func readBoundedDistributionFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("read built-in Recipe distribution %s: %w", filepath.Base(path), err)
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > maximumBuiltInRecipeAssetBytes {
		return nil, fmt.Errorf("%w: built-in Recipe distribution %s is not a bounded regular file", ErrInvalid, filepath.Base(path))
	}
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read built-in Recipe distribution %s: %w", filepath.Base(path), err)
	}
	return payload, nil
}

func ParseBuiltInRecipeDistribution(metadataBytes, configBytes []byte) (BuiltInRecipeDistribution, error) {
	if len(metadataBytes) == 0 || len(configBytes) == 0 ||
		len(metadataBytes) > maximumBuiltInRecipeAssetBytes || len(configBytes) > maximumBuiltInRecipeAssetBytes {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: built-in Recipe distribution files are invalid", ErrInvalid)
	}
	var metadata builtInRecipeMetadata
	if err := yaml.UnmarshalStrict(metadataBytes, &metadata); err != nil {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: decode built-in Recipe metadata: %w", ErrInvalid, err)
	}
	if metadata.SchemaVersion != "vllm-sr/recipe-metadata/v1" ||
		!resourceIDPattern.MatchString(metadata.ID) || !canonicalText(metadata.Name, 1, 128) ||
		!canonicalDistributionVersion(metadata.Version) {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: built-in Recipe metadata identity is invalid", ErrInvalid)
	}
	var document config.CanonicalConfig
	if err := yaml.UnmarshalStrict(configBytes, &document); err != nil {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: decode built-in Recipe config: %w", ErrInvalid, err)
	}
	if document.Version != "v0.4" {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: built-in Recipe distribution version must be v0.4", ErrInvalid)
	}
	if document.BillingCurrency != "" || len(document.Listeners) != 0 ||
		len(document.Models) != 0 || len(document.Entrypoints) != 0 || document.Global != nil {
		return BuiltInRecipeDistribution{}, fmt.Errorf(
			"%w: built-in Recipe distribution may contain only version and recipes", ErrInvalid,
		)
	}
	if len(document.Recipes) == 0 || len(document.Recipes) > 64 {
		return BuiltInRecipeDistribution{}, fmt.Errorf("%w: built-in Recipe distribution must contain between 1 and 64 Recipes", ErrInvalid)
	}

	digest := sha256.New()
	_, _ = digest.Write([]byte(builtInRecipeMetadataFile + "\x00"))
	_, _ = digest.Write(metadataBytes)
	_, _ = digest.Write([]byte("\x00" + builtInRecipeConfigFile + "\x00"))
	_, _ = digest.Write(configBytes)
	assetDigest := "sha256:" + hex.EncodeToString(digest.Sum(nil))

	result := BuiltInRecipeDistribution{
		ID: metadata.ID, Name: metadata.Name, Version: metadata.Version, AssetDigest: assetDigest,
	}
	seenSourceIDs := make(map[string]struct{}, len(document.Recipes))
	seenNames := make(map[string]struct{}, len(document.Recipes))
	for _, source := range document.Recipes {
		compiled, err := config.CompileStandaloneRoutingSnapshot(config.CanonicalConfig{
			Version: "v0.4",
			Recipes: []config.AuthoringRecipe{source},
		}, nil)
		if err != nil || len(compiled.Recipes) != 1 {
			return BuiltInRecipeDistribution{}, fmt.Errorf(
				"%w: compile built-in Recipe %q: %w", ErrInvalid, source.Name, err,
			)
		}
		compiledRecipe := compiled.Recipes[0]
		if _, duplicate := seenSourceIDs[compiledRecipe.ID]; duplicate {
			return BuiltInRecipeDistribution{}, fmt.Errorf(
				"%w: duplicate built-in Recipe source identity for %q", ErrInvalid, source.Name,
			)
		}
		seenSourceIDs[compiledRecipe.ID] = struct{}{}
		canonical, _, err := CompileRecipeDocument(compiledRecipe.ID, compiledRecipe.Document)
		if err != nil {
			return BuiltInRecipeDistribution{}, fmt.Errorf("built-in Recipe %q: %w", source.Name, err)
		}
		name := builtInRecipeName(metadata, source.Name)
		if _, duplicate := seenNames[name]; duplicate {
			return BuiltInRecipeDistribution{}, fmt.Errorf("%w: duplicate built-in Recipe name %q", ErrInvalid, name)
		}
		seenNames[name] = struct{}{}
		recipeDigest := sha256.Sum256(canonical)
		result.Recipes = append(result.Recipes, BuiltInRecipe{
			SourceID: compiledRecipe.ID, SourceRevision: compiledRecipe.Revision,
			Input:        RecipeInput{Name: name, Description: source.Description, Document: canonical},
			RecipeDigest: "sha256:" + hex.EncodeToString(recipeDigest[:]),
		})
	}
	sort.Slice(result.Recipes, func(i, j int) bool { return result.Recipes[i].SourceID < result.Recipes[j].SourceID })
	return result, nil
}

func canonicalDistributionVersion(value string) bool {
	if !canonicalText(value, 1, 64) {
		return false
	}
	for _, character := range value {
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '.' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func builtInRecipeName(metadata builtInRecipeMetadata, sourceName string) string {
	return fmt.Sprintf("%s %s / %s", metadata.Name, metadata.Version, sourceName)
}

// Validate proves that the distribution value can be installed. The asset
// parser calls this contract before returning, and durable stores repeat it at
// their trust boundary instead of accepting a caller-constructed value.
func (distribution BuiltInRecipeDistribution) Validate() error {
	if !resourceIDPattern.MatchString(distribution.ID) || !canonicalText(distribution.Name, 1, 128) ||
		!canonicalDistributionVersion(distribution.Version) || !digestPattern.MatchString(distribution.AssetDigest) ||
		len(distribution.Recipes) == 0 || len(distribution.Recipes) > 64 {
		return fmt.Errorf("%w: built-in Recipe distribution is invalid", ErrInvalid)
	}
	seenSources := make(map[string]struct{}, len(distribution.Recipes))
	seenNames := make(map[string]struct{}, len(distribution.Recipes))
	for _, recipe := range distribution.Recipes {
		if !resourceIDPattern.MatchString(recipe.SourceID) || recipe.SourceRevision <= 0 ||
			!digestPattern.MatchString(recipe.RecipeDigest) || recipe.Input.ID != "" {
			return fmt.Errorf("%w: built-in Recipe distribution member is invalid", ErrInvalid)
		}
		if _, duplicate := seenSources[recipe.SourceID]; duplicate {
			return fmt.Errorf("%w: built-in Recipe distribution source is duplicated", ErrInvalid)
		}
		if _, duplicate := seenNames[recipe.Input.Name]; duplicate {
			return fmt.Errorf("%w: built-in Recipe distribution name is duplicated", ErrInvalid)
		}
		seenSources[recipe.SourceID] = struct{}{}
		seenNames[recipe.Input.Name] = struct{}{}
		input := recipe.Input
		input.ID = distributionRecipeID(
			"00000000-0000-4000-8000-000000000000",
			distribution.ID, distribution.Version, recipe.SourceID,
		)
		compiled, err := compileRecipe(input, 1)
		if err != nil {
			return err
		}
		digest := sha256.Sum256(compiled.Document)
		if recipe.RecipeDigest != "sha256:"+hex.EncodeToString(digest[:]) {
			return fmt.Errorf("%w: built-in Recipe document digest is inconsistent", ErrInvalid)
		}
	}
	return nil
}

// RecipesForNamespace compiles deterministic ordinary Recipe resources for one
// Namespace. Namespace identity participates in each Recipe ID because the
// current durable Recipe key is global, while Recipe names and behavior remain
// identical across Namespaces.
func (distribution BuiltInRecipeDistribution) RecipesForNamespace(namespaceID string) ([]routingsnapshot.Recipe, error) {
	if err := distribution.Validate(); err != nil {
		return nil, err
	}
	if !canonicalUUIDText(namespaceID) {
		return nil, fmt.Errorf("%w: built-in Recipe namespace is invalid", ErrInvalid)
	}
	result := make([]routingsnapshot.Recipe, 0, len(distribution.Recipes))
	for _, member := range distribution.Recipes {
		input := member.Input
		input.ID = distributionRecipeID(namespaceID, distribution.ID, distribution.Version, member.SourceID)
		recipe, err := compileRecipe(input, 1)
		if err != nil {
			return nil, err
		}
		result = append(result, recipe)
	}
	return result, nil
}

func distributionRecipeID(namespaceID, distributionID, version, sourceRecipeID string) string {
	payload := strings.Join([]string{namespaceID, distributionID, version, sourceRecipeID}, "\x00")
	digest := sha256.Sum256([]byte(payload))
	return "rcp_" + hex.EncodeToString(digest[:16])
}

func canonicalUUIDText(value string) bool {
	if len(value) != 36 {
		return false
	}
	for index, character := range value {
		if index == 8 || index == 13 || index == 18 || index == 23 {
			if character != '-' {
				return false
			}
			continue
		}
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

type BuiltInRecipeInstallerOptions struct {
	Store             BuiltInRecipeStore
	Distribution      BuiltInRecipeDistribution
	ReconcileInterval time.Duration
	Now               func() time.Time
}

// BuiltInRecipeInstaller reconciles the Router distribution into every active
// namespace. PostgreSQL serializes competing replicas; this object holds no
// process-local installation authority.
type BuiltInRecipeInstaller struct {
	store        BuiltInRecipeStore
	distribution BuiltInRecipeDistribution
	interval     time.Duration
	now          func() time.Time

	mu            sync.RWMutex
	lastSuccessAt time.Time
	lastError     error
}

func NewBuiltInRecipeInstaller(options BuiltInRecipeInstallerOptions) (*BuiltInRecipeInstaller, error) {
	if options.Store == nil {
		return nil, errors.New("built-in Recipe installer store is required")
	}
	if err := options.Distribution.Validate(); err != nil {
		return nil, err
	}
	interval := options.ReconcileInterval
	if interval == 0 {
		interval = defaultBuiltInRecipeReconcileInterval
	}
	if interval < time.Second || interval > time.Hour {
		return nil, fmt.Errorf("%w: built-in Recipe reconcile interval is invalid", ErrInvalid)
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &BuiltInRecipeInstaller{
		store: options.Store, distribution: options.Distribution, interval: interval, now: now,
	}, nil
}

func (installer *BuiltInRecipeInstaller) Reconcile(ctx context.Context) error {
	if installer == nil || installer.store == nil {
		return errors.New("built-in Recipe installer is unavailable")
	}
	for {
		namespaces, reconcileErr := installer.store.PendingBuiltInRecipeNamespaces(
			ctx, installer.distribution, maximumBuiltInRecipeNamespaceBatch,
		)
		if reconcileErr != nil {
			return installer.record(reconcileErr)
		}
		if len(namespaces) == 0 {
			if err := installer.store.VerifyBuiltInRecipes(ctx, installer.distribution); err != nil {
				return installer.record(err)
			}
			return installer.record(nil)
		}
		for _, namespaceID := range namespaces {
			_, reconcileErr = installer.store.InstallBuiltInRecipes(ctx, namespaceID, installer.distribution, MutationContext{
				RequestID: "router-distribution:" + installer.distribution.AssetDigest,
				Reason:    "Install immutable built-in Recipe distribution " + installer.distribution.ID + "@" + installer.distribution.Version,
			})
			if reconcileErr != nil {
				return installer.record(reconcileErr)
			}
		}
	}
}

func (installer *BuiltInRecipeInstaller) record(err error) error {
	installer.mu.Lock()
	defer installer.mu.Unlock()
	installer.lastError = err
	if err == nil {
		installer.lastSuccessAt = installer.now().UTC()
	}
	return err
}

func (installer *BuiltInRecipeInstaller) Ready(ctx context.Context) error {
	if installer == nil || installer.store == nil {
		return errors.New("built-in Recipe installer is unavailable")
	}
	installer.mu.RLock()
	lastSuccessAt, lastError := installer.lastSuccessAt, installer.lastError
	installer.mu.RUnlock()
	if lastError != nil {
		return fmt.Errorf("built-in Recipe reconciliation failed: %w", lastError)
	}
	if lastSuccessAt.IsZero() {
		return errors.New("built-in Recipe reconciliation has not completed")
	}
	return nil
}

func (installer *BuiltInRecipeInstaller) Run(ctx context.Context) error {
	if err := installer.Reconcile(ctx); err != nil {
		return err
	}
	ticker := time.NewTicker(installer.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := installer.Reconcile(ctx); err != nil {
				return err
			}
		}
	}
}
