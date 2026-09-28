package recipe

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	packageRecordSchema           = "vllm-sr/recipe-package-record/v1"
	activePointerSchema           = "vllm-sr/recipe-active/v1"
	activationTxnSchema           = "vllm-sr/recipe-activation-transaction/v1"
	ActivationOperationActivate   = "activate"
	ActivationOperationDeactivate = "deactivate"
	ActivationTopologyNone        = "none"
	ActivationTopologyManaged     = "managed"
	ActivationFinalizing          = "finalizing"
	ActivationRollbackFinalizing  = "rollback_finalizing"
	defaultPackagePage            = 25
	maxPackagePage                = 100
	maxPackageQuery               = 256
)

type packageRecord struct {
	SchemaVersion   string           `json:"schema_version"`
	ID              string           `json:"id"`
	Name            string           `json:"name"`
	Version         string           `json:"version"`
	Description     string           `json:"description"`
	RecipeDigest    string           `json:"recipe_digest"`
	ConfigDigest    string           `json:"config_digest"`
	ArchiveSHA256   string           `json:"archive_sha256"`
	ArchiveVerified bool             `json:"archive_verified"`
	InstalledAt     time.Time        `json:"installed_at"`
	SourceURL       string           `json:"source_url"`
	Counts          Counts           `json:"counts"`
	Warnings        []PackageWarning `json:"warnings"`
}

type ActivePointer struct {
	SchemaVersion        string    `json:"schema_version"`
	RecipeDigest         string    `json:"recipe_digest"`
	ConfigDigest         string    `json:"config_digest"`
	RealizedConfigDigest string    `json:"realized_config_digest"`
	ActivatedAt          time.Time `json:"activated_at"`
}

type ActivationTransaction struct {
	SchemaVersion        string         `json:"schema_version"`
	State                string         `json:"state"`
	Operation            string         `json:"operation"`
	TopologyMode         string         `json:"topology_mode"`
	ID                   string         `json:"id"`
	TargetRecipeDigest   string         `json:"target_recipe_digest"`
	PreviousRecipeDigest string         `json:"previous_recipe_digest,omitempty"`
	PreviousPointer      *ActivePointer `json:"previous_pointer,omitempty"`
	PreviousConfigDigest string         `json:"previous_config_digest"`
	PreviousConfigBackup string         `json:"previous_config_backup"`
	CommitConfigDigest   string         `json:"commit_config_digest,omitempty"`
	StartedAt            time.Time      `json:"started_at"`
}

type StoreOptions struct {
	Root       string
	ConfigPath string
	HTTPClient *http.Client
	Resolver   IPResolver
	Now        func() time.Time
}

// Store owns one stack-scoped immutable package store. It never scans outside
// Root and indexes packages only through its controlled refs tree.
type Store struct {
	root       string
	configPath string
	httpClient *http.Client
	resolver   IPResolver
	now        func() time.Time
	mu         sync.Mutex
}

func NewStore(options StoreOptions) *Store {
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Store{
		root:       filepath.Clean(options.Root),
		configPath: filepath.Clean(options.ConfigPath),
		httpClient: options.HTTPClient,
		resolver:   options.Resolver,
		now:        now,
	}
}

func (s *Store) Root() string {
	return s.root
}

func (s *Store) ConfigPath() string {
	return s.configPath
}

func (s *Store) List(options PackageListOptions) (PackageList, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return PackageList{}, err
	}
	if err := normalizePackageListOptions(&options); err != nil {
		return PackageList{}, err
	}
	records, err := s.readRecords()
	if err != nil {
		return PackageList{}, err
	}
	active, state, _ := s.activationStatusLocked()
	activeDigest := activeDigestForState(active, state)
	records, err = s.includeDetachedActiveRecord(records, activeDigest)
	if err != nil {
		return PackageList{}, err
	}
	items := make([]PackageSummary, 0, len(records))
	for _, record := range records {
		if matchesPackageQuery(record, options.Query) {
			items = append(items, summaryFromRecord(record, activeDigest))
		}
	}
	total := len(items)
	totalPages := 0
	if total > 0 {
		totalPages = (total + options.PageSize - 1) / options.PageSize
	}
	start := total
	pageIndex := options.Page - 1
	if pageIndex <= total/options.PageSize {
		start = pageIndex * options.PageSize
	}
	end := min(start+options.PageSize, total)
	return PackageList{
		Items:           items[start:end],
		Page:            options.Page,
		PageSize:        options.PageSize,
		Total:           total,
		TotalPages:      totalPages,
		ActiveDigest:    activeDigest,
		ActivationState: state,
	}, nil
}

func (s *Store) Package(recipeDigest string) (PackageSummary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return PackageSummary{}, err
	}
	record, err := s.findRecordByDigest(recipeDigest)
	if err != nil {
		return PackageSummary{}, err
	}
	active, state, _ := s.activationStatusLocked()
	return summaryFromRecord(record, activeDigestForState(active, state)), nil
}

func (s *Store) ObjectDirectory(recipeDigest string) (string, error) {
	normalized, err := normalizeRecipeDigest(recipeDigest)
	if err != nil {
		return "", err
	}
	return filepath.Join(s.root, "objects", "sha256", strings.TrimPrefix(normalized, "sha256:")), nil
}

func (s *Store) ReadActivePointer() (ActivePointer, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readActivePointer()
}

func (s *Store) ActivationStatus() (ActivePointer, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return ActivePointer{}, ActivationInconsistent, err
	}
	return s.activationStatusLocked()
}

func (s *Store) activationStatusLocked() (ActivePointer, string, error) {
	transaction, txnErr := s.readTransaction()
	if txnErr == nil {
		state := ActivationRecovering
		if transaction.State == ActivationInconsistent {
			state = ActivationInconsistent
		}
		active, _ := s.readActivePointer()
		return active, state, nil
	}
	if !errors.Is(txnErr, os.ErrNotExist) {
		return ActivePointer{}, ActivationInconsistent, txnErr
	}
	active, err := s.readActivePointer()
	if errors.Is(err, os.ErrNotExist) {
		return ActivePointer{}, ActivationNone, nil
	}
	if err != nil {
		return ActivePointer{}, ActivationInconsistent, err
	}
	if err := s.validateActivePointer(active); err != nil {
		return active, ActivationInconsistent, err
	}
	return active, ActivationActive, nil
}

func (s *Store) validateActivePointer(pointer ActivePointer) error {
	objectDir, err := s.ObjectDirectory(pointer.RecipeDigest)
	if err != nil {
		return err
	}
	files, err := readRecipeObject(objectDir)
	if err != nil {
		return err
	}
	if digestRecipe(files) != pointer.RecipeDigest || digestBytes(files["config"]) != pointer.ConfigDigest {
		return errors.New("active package content does not match its pointer")
	}
	config, err := readBoundedFile(s.configPath, maxConfigBytes)
	if err != nil {
		return err
	}
	if digestBytes(config) != pointer.RealizedConfigDigest {
		return errors.New("active runtime config drifted from the activated package")
	}
	return nil
}

func (s *Store) readActivePointer() (ActivePointer, error) {
	var pointer ActivePointer
	if err := readStrictJSONFile(filepath.Join(s.root, "active.json"), 64<<10, &pointer); err != nil {
		return ActivePointer{}, err
	}
	if pointer.SchemaVersion != activePointerSchema {
		return ActivePointer{}, errors.New("unsupported active pointer schema")
	}
	if _, err := normalizeRecipeDigest(pointer.RecipeDigest); err != nil {
		return ActivePointer{}, err
	}
	if !validDigest(pointer.ConfigDigest) || !validDigest(pointer.RealizedConfigDigest) {
		return ActivePointer{}, errors.New("active pointer contains an invalid config digest")
	}
	return pointer, nil
}

func (s *Store) readRecords() ([]packageRecord, error) {
	refsRoot := filepath.Join(s.root, "refs")
	ids, err := readDirectories(refsRoot)
	if err != nil {
		return nil, err
	}
	records := []packageRecord{}
	for _, id := range ids {
		idRecords, err := readRecordsForID(refsRoot, id)
		if err != nil {
			return nil, err
		}
		records = append(records, idRecords...)
	}
	sort.Slice(records, func(i, j int) bool {
		left, right := records[i], records[j]
		if left.ID != right.ID {
			return left.ID < right.ID
		}
		if left.Version != right.Version {
			return left.Version < right.Version
		}
		return left.RecipeDigest < right.RecipeDigest
	})
	return records, nil
}

func readRecordsForID(refsRoot, id string) ([]packageRecord, error) {
	entries, err := os.ReadDir(filepath.Join(refsRoot, id))
	if err != nil {
		return nil, err
	}
	records := make([]packageRecord, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || filepath.Ext(entry.Name()) != ".json" {
			return nil, errors.New("package refs contain an invalid entry")
		}
		var record packageRecord
		path := filepath.Join(refsRoot, id, entry.Name())
		if err := readStrictJSONFile(path, 1<<20, &record); err != nil {
			return nil, err
		}
		if err := validateRecord(record, id); err != nil {
			return nil, err
		}
		if entry.Name() != record.Version+".json" {
			return nil, errors.New("package ref filename does not match its version")
		}
		records = append(records, record)
	}
	return records, nil
}

func (s *Store) findRecordByDigest(raw string) (packageRecord, error) {
	digest, err := normalizeRecipeDigest(raw)
	if err != nil {
		return packageRecord{}, err
	}
	records, err := s.readRecords()
	if err != nil {
		return packageRecord{}, err
	}
	for _, record := range records {
		if record.RecipeDigest == digest {
			return record, nil
		}
	}
	historical, historicalErr := s.readDigestRecord(digest)
	if historicalErr == nil {
		return historical, nil
	}
	if !errors.Is(historicalErr, os.ErrNotExist) {
		return packageRecord{}, historicalErr
	}
	return packageRecord{}, wrapPackageError(ErrorPackageNotFound, http.StatusNotFound, "Recipe package was not found.", os.ErrNotExist)
}

func (s *Store) includeDetachedActiveRecord(records []packageRecord, activeDigest string) ([]packageRecord, error) {
	if activeDigest == "" {
		return records, nil
	}
	for _, record := range records {
		if record.RecipeDigest == activeDigest {
			return records, nil
		}
	}
	activeRecord, err := s.readDigestRecord(activeDigest)
	if err != nil {
		return nil, err
	}
	if activeRecord.RecipeDigest != activeDigest {
		return nil, errors.New("active package record does not match its pointer")
	}
	records = append(records, activeRecord)
	sort.Slice(records, func(i, j int) bool {
		left, right := records[i], records[j]
		if left.ID != right.ID {
			return left.ID < right.ID
		}
		if left.Version != right.Version {
			return left.Version < right.Version
		}
		return left.RecipeDigest < right.RecipeDigest
	})
	return records, nil
}

func (s *Store) readDigestRecord(digest string) (packageRecord, error) {
	var record packageRecord
	if !validDigest(digest) {
		return packageRecord{}, errors.New("invalid historical package digest")
	}
	path := filepath.Join(s.root, "records", "sha256", strings.TrimPrefix(digest, "sha256:")+".json")
	if err := readStrictJSONFile(path, 1<<20, &record); err != nil {
		return packageRecord{}, err
	}
	if err := validateRecord(record, record.ID); err != nil || record.RecipeDigest != digest {
		if err == nil {
			err = errors.New("historical package record digest mismatch")
		}
		return packageRecord{}, err
	}
	return record, nil
}

func (s *Store) ensureLayout() error {
	if s.root == "." || s.root == string(filepath.Separator) || strings.TrimSpace(s.root) == "" {
		return errors.New("recipe store root is not configured")
	}
	for _, path := range []string{
		s.root,
		filepath.Join(s.root, "objects"),
		filepath.Join(s.root, "objects", "sha256"),
		filepath.Join(s.root, "refs"),
		filepath.Join(s.root, "records"),
		filepath.Join(s.root, "records", "sha256"),
		filepath.Join(s.root, ".staging"),
		filepath.Join(s.root, "transactions"),
		filepath.Join(s.root, "credentials"),
		filepath.Join(s.root, "source-baselines"),
	} {
		if err := ensureRealDirectory(path); err != nil {
			return err
		}
	}
	return nil
}

func ensureRealDirectory(path string) error {
	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("recipe store path must be a real directory")
	}
	return nil
}

func readDirectories(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	result := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || !metadataIDPattern.MatchString(entry.Name()) {
			return nil, errors.New("package refs contain an invalid directory")
		}
		result = append(result, entry.Name())
	}
	sort.Strings(result)
	return result, nil
}

func readStrictJSONFile(path string, maxBytes int64, target any) error {
	data, err := readBoundedFile(path, maxBytes)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return errors.New("JSON file must contain one document")
	}
	return nil
}

func writeJSONAtomically(path string, value any, mode os.FileMode) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeFileAtomically(path, data, mode)
}

func writeFileAtomically(path string, data []byte, mode os.FileMode) error {
	parent := filepath.Dir(path)
	if err := ensureRealDirectory(parent); err != nil {
		return err
	}
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		return err
	}
	temp := filepath.Join(parent, "."+filepath.Base(path)+"."+hex.EncodeToString(random)+".tmp")
	file, err := os.OpenFile(temp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	cleanup := func() { _ = os.Remove(temp) }
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		cleanup()
		return err
	}
	if err := os.Rename(temp, path); err != nil {
		cleanup()
		return err
	}
	return syncDirectory(parent)
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	return directory.Sync()
}

func validateRecord(record packageRecord, directoryID string) error {
	if record.SchemaVersion != packageRecordSchema || record.ID != directoryID || !metadataIDPattern.MatchString(record.ID) {
		return errors.New("package ref has invalid identity")
	}
	if !semverPattern.MatchString(record.Version) || !validDigest(record.RecipeDigest) || !validDigest(record.ConfigDigest) || !validDigest(record.ArchiveSHA256) {
		return errors.New("package ref has invalid version or digest")
	}
	return nil
}

func normalizePackageListOptions(options *PackageListOptions) error {
	if options.Page == 0 {
		options.Page = 1
	}
	if options.PageSize == 0 {
		options.PageSize = defaultPackagePage
	}
	if options.Page < 1 || options.PageSize < 1 || options.PageSize > maxPackagePage {
		return wrapPackageError(ErrorInvalidRequest, http.StatusBadRequest, "Invalid package pagination.", nil)
	}
	options.Query = strings.TrimSpace(options.Query)
	if len(options.Query) > maxPackageQuery {
		return wrapPackageError(ErrorInvalidRequest, http.StatusBadRequest, "Package search is too long.", nil)
	}
	return nil
}

func matchesPackageQuery(record packageRecord, query string) bool {
	if query == "" {
		return true
	}
	haystack := strings.ToLower(strings.Join([]string{
		record.ID,
		record.Name,
		record.Version,
		record.Description,
		record.RecipeDigest,
		record.ConfigDigest,
	}, " "))
	return strings.Contains(haystack, strings.ToLower(query))
}

func summaryFromRecord(record packageRecord, activeDigest string) PackageSummary {
	warnings := record.Warnings
	if warnings == nil {
		warnings = []PackageWarning{}
	}
	return PackageSummary{
		ID:              record.ID,
		Name:            record.Name,
		Version:         record.Version,
		Description:     record.Description,
		RecipeDigest:    record.RecipeDigest,
		ConfigDigest:    record.ConfigDigest,
		ArchiveSHA256:   record.ArchiveSHA256,
		ArchiveVerified: record.ArchiveVerified,
		InstalledAt:     record.InstalledAt,
		Active:          record.RecipeDigest == activeDigest,
		Counts:          record.Counts,
		Warnings:        warnings,
	}
}

func activeDigestForState(pointer ActivePointer, state string) string {
	if state == ActivationActive {
		return pointer.RecipeDigest
	}
	return ""
}

func validDigest(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil && value == strings.ToLower(value)
}

func normalizeRecipeDigest(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if !validDigest(value) {
		return "", wrapPackageError(ErrorInvalidRequest, http.StatusBadRequest, "recipe_digest must be a SHA-256 digest.", nil)
	}
	return value, nil
}

func readRecipeObject(directory string) (map[string][]byte, error) {
	info, err := os.Lstat(directory)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, errors.New("stored Recipe object must be a real directory")
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	if len(entries) != len(recipeFiles) {
		return nil, errors.New("stored Recipe object does not contain exactly five files")
	}
	files := make(map[string][]byte, len(recipeFiles))
	for _, file := range recipeFiles {
		data, readErr := readBoundedFile(filepath.Join(directory, file.name), file.maxBytes)
		if readErr != nil {
			return nil, readErr
		}
		files[file.key] = data
	}
	return files, nil
}

func randomID() (string, error) {
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		return "", err
	}
	return hex.EncodeToString(value), nil
}

func recordRefPath(root string, record packageRecord) string {
	return filepath.Join(root, "refs", record.ID, record.Version+".json")
}
