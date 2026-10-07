package configsnapshot

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot/historylock"
)

const (
	recordPrefix         = "config."
	recordDocumentSuffix = ".yaml"
	recordSourceSuffix   = ".source"
	recordSnapshotSuffix = ".snapshot.json"
)

// DirStore keeps history records as files in one directory, in the layout the
// management API and the Dashboard have used for their backups:
// config.<id>.yaml holds a record's document, config.<id>.source its source,
// and config.<id>.snapshot.json the rest of it. A document without a snapshot
// file, such as a backup the Dashboard wrote, loads as a record of version 0.
type DirStore struct {
	dir string
}

// NewDirStore returns the store of dir. The directory is created, private to
// its owner, on the first save.
func NewDirStore(dir string) *DirStore { return &DirStore{dir: dir} }

// Load returns every record in the directory, or none when it does not exist.
// A record that cannot be read is skipped and reported in the error.
func (s *DirStore) Load() ([]Record, error) {
	entries, err := os.ReadDir(s.dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read configuration history: %w", err)
	}
	ids := make(map[string]bool)
	for _, entry := range entries {
		if id, ok := recordFileID(entry.Name()); ok && !entry.IsDir() {
			ids[id] = true
		}
	}
	records := make([]Record, 0, len(ids))
	var errs []error
	for _, id := range sortedKeys(ids) {
		record, loadErr := s.load(id)
		if loadErr != nil {
			errs = append(errs, loadErr)
			continue
		}
		records = append(records, record)
	}
	return records, errors.Join(errs...)
}

func (s *DirStore) load(id string) (Record, error) {
	record := Record{ID: id}
	document, docErr := os.ReadFile(s.path(id, recordDocumentSuffix))
	if docErr == nil {
		record.Document = document
	} else if !errors.Is(docErr, os.ErrNotExist) {
		return Record{}, fmt.Errorf("read configuration record %s: %w", id, docErr)
	}
	meta, metaErr := os.ReadFile(s.path(id, recordSnapshotSuffix))
	switch {
	case metaErr == nil:
		if err := json.Unmarshal(meta, &record); err != nil {
			return Record{}, fmt.Errorf("read configuration record %s: %w", id, err)
		}
		record.ID, record.Document = id, document
	case !errors.Is(metaErr, os.ErrNotExist):
		return Record{}, fmt.Errorf("read configuration record %s: %w", id, metaErr)
	case record.Document == nil:
		return Record{}, fmt.Errorf("configuration record %s has neither a document nor snapshot metadata", id)
	default:
		record.Hash = documentDigest(record.Document)
		record.Origin.Source = s.legacySource(id)
		record.RecordedAt = legacyRecordTime(id, s.path(id, recordDocumentSuffix))
	}
	return record, nil
}

// legacySource reads the source a pre-snapshot backup recorded, if any.
func (s *DirStore) legacySource(id string) Source {
	data, err := os.ReadFile(s.path(id, recordSourceSuffix))
	if err != nil {
		return ""
	}
	return Source(strings.TrimSpace(string(data)))
}

// legacyRecordTime is the time an ID names, or the document's modification
// time when the ID is in another layout.
func legacyRecordTime(id, documentPath string) time.Time {
	if len(id) >= len(recordIDLayout) {
		if at, err := time.ParseInLocation(recordIDLayout, id[:len(recordIDLayout)], time.Local); err == nil {
			return at.UTC()
		}
	}
	if info, err := os.Stat(documentPath); err == nil {
		return info.ModTime().UTC()
	}
	return time.Time{}
}

// Save writes r under its ID, or under the next ID when another writer of the
// directory took it. Files are created exclusively, so it never replaces
// another writer's record.
func (s *DirStore) Save(r Record) (Record, error) {
	if !recordIDPattern.MatchString(r.ID) {
		return Record{}, fmt.Errorf("record configuration version: invalid record ID %q", r.ID)
	}
	if err := ensurePrivateDir(s.dir); err != nil {
		return Record{}, fmt.Errorf("prepare configuration history: %w", err)
	}
	for ; ; r.ID = nextRecordID(r.ID) {
		if s.taken(r.ID) {
			continue
		}
		err := s.write(r)
		if errors.Is(err, os.ErrExist) {
			continue
		}
		if err != nil {
			return Record{}, fmt.Errorf("record configuration version: %w", err)
		}
		return r, nil
	}
}

// taken reports whether a record with id exists.
func (s *DirStore) taken(id string) bool {
	for _, suffix := range []string{recordDocumentSuffix, recordSnapshotSuffix} {
		if _, err := os.Lstat(s.path(id, suffix)); !errors.Is(err, os.ErrNotExist) {
			return true
		}
	}
	return false
}

// write creates r's files, removing those it created when one fails.
func (s *DirStore) write(r Record) error {
	meta, err := json.Marshal(r)
	if err != nil {
		return err
	}
	files := []struct {
		suffix string
		data   []byte
	}{
		{recordSnapshotSuffix, meta},
		{recordDocumentSuffix, r.Document},
		{recordSourceSuffix, []byte(string(r.Origin.Source) + "\n")},
	}
	var written []string
	for _, file := range files {
		if file.suffix == recordDocumentSuffix && r.Document == nil {
			continue
		}
		path := s.path(r.ID, file.suffix)
		if err := writePrivateFile(path, file.data); err != nil {
			for _, done := range written {
				_ = os.Remove(done)
			}
			return err
		}
		written = append(written, path)
	}
	return nil
}

// Lock holds the directory's lock, so that its writers take turns in any
// process.
func (s *DirStore) Lock() (func(), error) { return historylock.Lock(s.dir) }

// Remove deletes the files of the record with id.
func (s *DirStore) Remove(id string) error {
	var errs []error
	for _, suffix := range []string{recordDocumentSuffix, recordSourceSuffix, recordSnapshotSuffix} {
		if err := os.Remove(s.path(id, suffix)); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (s *DirStore) path(id, suffix string) string {
	return filepath.Join(s.dir, recordPrefix+id+suffix)
}

// recordFileID returns the record ID a history file belongs to.
func recordFileID(name string) (string, bool) {
	if !strings.HasPrefix(name, recordPrefix) {
		return "", false
	}
	for _, suffix := range []string{recordSnapshotSuffix, recordDocumentSuffix} {
		if strings.HasSuffix(name, suffix) {
			id := strings.TrimSuffix(strings.TrimPrefix(name, recordPrefix), suffix)
			return id, id != "" && !strings.ContainsAny(id, `/\`)
		}
	}
	return "", false
}

func ensurePrivateDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	return os.Chmod(dir, 0o700)
}

// writePrivateFile creates path for its owner only and syncs it; it fails with
// os.ErrExist when the file exists.
func writePrivateFile(path string, data []byte) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(data)
	syncErr := file.Sync()
	closeErr := file.Close()
	if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}
