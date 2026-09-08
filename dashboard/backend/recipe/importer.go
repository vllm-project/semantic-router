package recipe

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

const (
	maxExpandedBytes int64 = maxMetadataBytes + maxConfigBytes + maxProbesBytes + maxDSLBytes + maxREADMEBytes
	// An exact-five ZIP can be stored rather than compressed, so the wire bound
	// must cover the full per-file expansion budget plus bounded ZIP metadata.
	maxArchiveBytes int64 = maxExpandedBytes + (1 << 20)
	maxRedirects          = 5
)

type IPResolver interface {
	LookupNetIP(context.Context, string, string) ([]netip.Addr, error)
}

type extractedPackage struct {
	files           map[string][]byte
	archiveDigest   string
	archiveVerified bool
	sanitizedSource string
}

func (s *Store) Import(ctx context.Context, request ImportRequest) (PackageSummary, bool, error) {
	if err := s.prepareLayout(); err != nil {
		return PackageSummary{}, false, err
	}
	extracted, err := s.fetchPackage(ctx, request)
	if err != nil {
		return PackageSummary{}, false, err
	}
	return s.installExtracted(extracted)
}

func (s *Store) prepareLayout() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package store is unavailable.", err)
	}
	return nil
}

func (s *Store) fetchPackage(ctx context.Context, request ImportRequest) (extractedPackage, error) {
	remoteURL, err := normalizeRemotePackageURL(request.URL)
	if err != nil {
		return extractedPackage{}, err
	}
	expected, err := normalizeExpectedArchiveDigest(request.ExpectedArchiveSHA256)
	if err != nil {
		return extractedPackage{}, err
	}
	client := s.httpClient
	if client == nil {
		client = newPackageHTTPClient(s.resolver)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, remoteURL.String(), nil)
	if err != nil {
		return extractedPackage{}, wrapPackageError(ErrorInvalidURL, http.StatusBadRequest, "Recipe package URL is invalid.", err)
	}
	req.Header.Set("Accept", "application/zip, application/octet-stream")
	req.Header.Set("Accept-Encoding", "identity")
	response, err := client.Do(req)
	if err != nil {
		return extractedPackage{}, classifyDownloadError(err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.Request == nil || response.Request.URL == nil {
		return extractedPackage{}, wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download returned an invalid response.", nil)
	}
	if _, redirectErr := normalizeRemotePackageURL(response.Request.URL.String()); redirectErr != nil {
		return extractedPackage{}, redirectErr
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return extractedPackage{}, wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download failed.", fmt.Errorf("HTTP %d", response.StatusCode))
	}
	if response.ContentLength > maxArchiveBytes {
		return extractedPackage{}, wrapPackageError(ErrorDownloadTooLarge, http.StatusRequestEntityTooLarge, "Recipe package exceeds the download limit.", nil)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxArchiveBytes+1))
	if err != nil {
		return extractedPackage{}, wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download failed.", err)
	}
	if int64(len(body)) > maxArchiveBytes {
		return extractedPackage{}, wrapPackageError(ErrorDownloadTooLarge, http.StatusRequestEntityTooLarge, "Recipe package exceeds the download limit.", nil)
	}
	archiveHash := sha256.Sum256(body)
	archiveDigest := "sha256:" + hex.EncodeToString(archiveHash[:])
	if expected != "" && archiveDigest != expected {
		return extractedPackage{}, wrapPackageError(ErrorArchiveDigestMismatch, http.StatusUnprocessableEntity, "Recipe package archive digest does not match the expected SHA-256.", nil)
	}
	files, err := extractRecipeZIP(body)
	if err != nil {
		return extractedPackage{}, err
	}
	return extractedPackage{
		files:           files,
		archiveDigest:   archiveDigest,
		archiveVerified: expected != "",
		sanitizedSource: sanitizeSourceURL(response.Request.URL),
	}, nil
}

func (s *Store) installExtracted(extracted extractedPackage) (PackageSummary, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return PackageSummary{}, false, wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package store is unavailable.", err)
	}
	staging, err := os.MkdirTemp(filepath.Join(s.root, ".staging"), "import-")
	if err != nil {
		return PackageSummary{}, false, wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package could not be staged.", err)
	}
	defer func() { _ = os.RemoveAll(staging) }()
	if stagingErr := writeStagedFiles(staging, extracted.files); stagingErr != nil {
		return PackageSummary{}, false, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package could not be staged safely.", stagingErr)
	}
	validation, err := validatePackageDirectory(staging)
	if err != nil {
		return PackageSummary{}, false, err
	}
	recipeDigest := digestRecipe(extracted.files)
	record := packageRecord{
		SchemaVersion:   packageRecordSchema,
		ID:              validation.metadata.ID,
		Name:            validation.metadata.Name,
		Version:         validation.metadata.Version,
		Description:     validation.metadata.Description,
		RecipeDigest:    recipeDigest,
		ConfigDigest:    digestBytes(extracted.files["config"]),
		ArchiveSHA256:   extracted.archiveDigest,
		ArchiveVerified: extracted.archiveVerified,
		InstalledAt:     s.now().UTC(),
		SourceURL:       extracted.sanitizedSource,
		Counts:          validation.counts,
		Warnings:        validation.warnings,
	}
	existing, found, referenceErr := s.readVersionRef(record)
	if referenceErr != nil {
		return PackageSummary{}, false, referenceErr
	}
	if found {
		if existing.RecipeDigest == record.RecipeDigest {
			active, state, _ := s.activationStatusLocked()
			return summaryFromRecord(existing, activeDigestForState(active, state)), false, nil
		}
	}
	objectDirectory, _ := s.ObjectDirectory(recipeDigest)
	createdObject, err := promoteStagingDirectory(staging, objectDirectory, extracted.files)
	if err != nil {
		return PackageSummary{}, false, wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package could not be installed.", err)
	}
	if err := s.writeDigestRecord(record); err != nil {
		if createdObject {
			_ = os.RemoveAll(objectDirectory)
		}
		return PackageSummary{}, false, wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package history could not be updated.", err)
	}
	if err := writeJSONAtomically(recordRefPath(s.root, record), record, 0o600); err != nil {
		return PackageSummary{}, false, wrapPackageError(ErrorActivationFailed, http.StatusInternalServerError, "Recipe package index could not be updated.", err)
	}
	active, state, _ := s.activationStatusLocked()
	return summaryFromRecord(record, activeDigestForState(active, state)), true, nil
}

func (s *Store) writeDigestRecord(record packageRecord) error {
	path := filepath.Join(s.root, "records", "sha256", strings.TrimPrefix(record.RecipeDigest, "sha256:")+".json")
	if existing, err := s.readDigestRecord(record.RecipeDigest); err == nil {
		if existing.RecipeDigest != record.RecipeDigest || existing.ID != record.ID || existing.Version != record.Version {
			return errors.New("historical Recipe package record is inconsistent")
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeJSONAtomically(path, record, 0o600)
}

func (s *Store) readVersionRef(record packageRecord) (packageRecord, bool, error) {
	path := recordRefPath(s.root, record)
	var existing packageRecord
	err := readStrictJSONFile(path, 1<<20, &existing)
	if errors.Is(err, os.ErrNotExist) {
		return packageRecord{}, false, nil
	}
	if err != nil {
		return packageRecord{}, false, err
	}
	if err := validateRecord(existing, record.ID); err != nil {
		return packageRecord{}, false, err
	}
	return existing, true, nil
}

func writeStagedFiles(directory string, files map[string][]byte) error {
	for _, spec := range recipeFiles {
		data, ok := files[spec.key]
		if !ok {
			return fmt.Errorf("missing %s", spec.name)
		}
		path := filepath.Join(directory, spec.name)
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			return err
		}
		_, writeErr := file.Write(data)
		if writeErr == nil {
			writeErr = file.Sync()
		}
		if closeErr := file.Close(); writeErr == nil {
			writeErr = closeErr
		}
		if writeErr != nil {
			return writeErr
		}
	}
	return syncDirectory(directory)
}

func promoteStagingDirectory(staging, target string, expected map[string][]byte) (bool, error) {
	if err := ensureRealDirectory(filepath.Dir(target)); err != nil {
		return false, err
	}
	if _, err := os.Lstat(target); err == nil {
		files, readErr := readRecipeObject(target)
		if readErr != nil || digestRecipe(files) != digestRecipe(expected) {
			return false, errors.New("content-addressed Recipe object is inconsistent")
		}
		return false, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}
	if err := os.Rename(staging, target); err != nil {
		return false, err
	}
	if err := syncDirectory(filepath.Dir(target)); err != nil {
		return true, err
	}
	return true, nil
}

func extractRecipeZIP(data []byte) (map[string][]byte, error) {
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package is not a valid ZIP archive.", err)
	}
	if len(reader.File) != len(recipeFiles) {
		return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package must contain exactly the five Recipe files at its root.", nil)
	}
	allowed := make(map[string]struct{}, len(recipeFiles))
	limits := make(map[string]int64, len(recipeFiles))
	keys := make(map[string]string, len(recipeFiles))
	for _, file := range recipeFiles {
		allowed[file.name] = struct{}{}
		limits[file.name] = file.maxBytes
		keys[file.name] = file.key
	}
	seen := map[string]struct{}{}
	files := make(map[string][]byte, len(recipeFiles))
	var expanded uint64
	for _, entry := range reader.File {
		name := entry.Name
		lower := strings.ToLower(name)
		_, expected := allowed[name]
		_, duplicate := seen[lower]
		invalidPath := entry.NonUTF8 || !utf8.ValidString(name) || strings.ContainsAny(name, "\\\x00") || filepath.Base(name) != name || name == "." || name == ".."
		invalidMode := entry.FileInfo().IsDir() || entry.Mode()&os.ModeSymlink != 0 || !entry.Mode().IsRegular()
		unsupported := entry.Flags&0x1 != 0 || (entry.Method != zip.Store && entry.Method != zip.Deflate)
		if !expected || duplicate || invalidPath || invalidMode || unsupported {
			return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package contains an unsafe or unsupported ZIP entry.", nil)
		}
		seen[lower] = struct{}{}
		limit := limits[name]
		// #nosec G115 -- all Recipe file limits are positive bounded constants.
		if entry.UncompressedSize64 > uint64(limit) || expanded+entry.UncompressedSize64 > uint64(maxExpandedBytes) {
			return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package expanded content exceeds its size limit.", nil)
		}
		content, readErr := readZIPEntry(entry, limit)
		if readErr != nil {
			return nil, readErr
		}
		files[keys[name]] = content
		expanded += uint64(len(content))
	}
	return files, nil
}

func readZIPEntry(entry *zip.File, limit int64) ([]byte, error) {
	reader, err := entry.Open()
	if err != nil {
		return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package ZIP entry could not be opened.", err)
	}
	defer func() { _ = reader.Close() }()
	content, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil || int64(len(content)) > limit {
		return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package file exceeds its size limit.", err)
	}
	if !utf8.Valid(content) {
		return nil, wrapPackageError(ErrorInvalidArchive, http.StatusUnprocessableEntity, "Recipe package files must be valid UTF-8.", nil)
	}
	return content, nil
}

func normalizeExpectedArchiveDigest(raw string) (string, error) {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return "", nil
	}
	if !strings.HasPrefix(value, "sha256:") {
		value = "sha256:" + value
	}
	if !validDigest(value) {
		return "", wrapPackageError(ErrorInvalidRequest, http.StatusBadRequest, "expected_archive_sha256 must be a SHA-256 digest.", nil)
	}
	return value, nil
}

func normalizeRemotePackageURL(raw string) (*url.URL, error) {
	parsed, err := packagePolicy(nil).ValidateURL(raw)
	if err != nil {
		return nil, classifyFetchPolicyError(err)
	}
	return parsed, nil
}

func sanitizeSourceURL(source *url.URL) string {
	if source == nil {
		return ""
	}
	sanitized := *source
	sanitized.User = nil
	sanitized.RawQuery = ""
	sanitized.ForceQuery = false
	sanitized.Fragment = ""
	return sanitized.String()
}

func classifyDownloadError(err error) error {
	var packageErr *PackageError
	if errors.As(err, &packageErr) {
		return err
	}
	if mapped := classifyFetchPolicyError(err); mapped != err {
		return mapped
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download was cancelled or timed out.", err)
	}
	return wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download failed.", err)
}

// packagePolicy is the shared outbound policy, tightened for package
// downloads: HTTPS only, and a longer deadline because an archive is larger
// than a page.
func packagePolicy(resolver IPResolver) safefetch.Policy {
	policy := safefetch.DefaultPolicy().
		WithSchemes("https").
		WithTimeout(60 * time.Second)
	policy.MaxRedirects = maxRedirects
	if resolver != nil {
		policy = policy.WithResolver(resolver)
	}
	return policy
}

func newPackageHTTPClient(resolver IPResolver) *http.Client {
	return packagePolicy(resolver).NewClient()
}

// classifyFetchPolicyError maps a shared-policy refusal onto this package's
// error codes, so the importer's API contract is unchanged by the move.
func classifyFetchPolicyError(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, safefetch.ErrDestinationForbidden):
		return wrapPackageError(ErrorSourceForbidden, http.StatusBadRequest, "Recipe package URL resolves to a non-public address.", nil)
	case errors.Is(err, safefetch.ErrSchemeNotAllowed):
		return wrapPackageError(ErrorInsecureURL, http.StatusBadRequest, "Recipe package URL must use HTTPS.", nil)
	case errors.Is(err, safefetch.ErrInvalidURL):
		return wrapPackageError(ErrorInvalidURL, http.StatusBadRequest, "Recipe package URL is invalid.", nil)
	case errors.Is(err, safefetch.ErrTooManyRedirects):
		return wrapPackageError(ErrorDownloadFailed, http.StatusBadGateway, "Recipe package download used too many redirects.", nil)
	default:
		return err
	}
}
