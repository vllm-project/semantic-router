package configsnapshot

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"sync"
	"time"
)

// DefaultHistoryLimit is how many records a history keeps by default: as many
// as the management API kept backups before the history existed.
const DefaultHistoryLimit = 10

// recordIDLayout names records by the local time they were written, as the
// management API named its backups. A record written in the same second as an
// earlier one takes a numeric suffix.
const recordIDLayout = "20060102-150405"

// recordIDPattern matches the IDs a History assigns.
var recordIDPattern = regexp.MustCompile(`^([0-9]{8}-[0-9]{6})(?:-([0-9]{3,9}))?$`)

// Record is one entry of the configuration history: a snapshot that activated,
// or a document the management API preserved before it replaced it, which
// never activated as recorded (Version 0).
type Record struct {
	// ID names the record. IDs sort in the order records were written.
	ID      string `json:"id"`
	Version uint64 `json:"version,omitempty"`
	Hash    string `json:"hash"`
	Origin  Origin `json:"origin"`
	// RecordedAt is when the snapshot activated or the document was kept.
	RecordedAt time.Time `json:"recorded_at"`
	// Document is the recorded document; nil when its source had none.
	Document []byte `json:"-"`
}

// HistoryStore persists history records.
type HistoryStore interface {
	// Load returns every stored record.
	Load() ([]Record, error)
	// Save stores a record under its ID, or under the next free ID when
	// another writer took that one, and returns it with the ID it used.
	Save(Record) (Record, error)
	// Remove deletes the record with id.
	Remove(id string) error
}

// StoreLocker is a HistoryStore that several writers share, in this process
// or others. Lock holds it exclusively until unlock is called.
type StoreLocker interface {
	Lock() (unlock func(), err error)
}

// History keeps the most recent configuration records. It is safe for
// concurrent use.
type History struct {
	limit int
	store HistoryStore

	// writes orders this process's writes; the store's lock orders them
	// with other processes'.
	writes sync.Mutex

	mu      sync.Mutex
	records []Record // oldest first
	// unsaved names the records the store failed to persist, which only
	// this history holds.
	unsaved map[string]bool
	// lastID is the greatest ID the history assigned or loaded; new IDs
	// follow it, so no ID names two records.
	lastID string
}

// HistoryWrite is an exclusive write to a history: every other writer of
// its store waits until it ends, and it sees what they recorded before it
// began.
type HistoryWrite struct {
	h      *History
	unlock func()
	// err is why the store could not be locked or read; the write goes on
	// with what the history holds.
	err error
}

// Begin starts an exclusive write. End must be called.
func (h *History) Begin() *HistoryWrite {
	h.writes.Lock()
	w := &HistoryWrite{h: h, unlock: func() {}}
	if h.store == nil {
		return w
	}
	if locker, ok := h.store.(StoreLocker); ok {
		unlock, err := locker.Lock()
		if err != nil {
			w.err = fmt.Errorf("lock the configuration history: %w", err)
		} else {
			w.unlock = unlock
		}
	}
	w.err = errors.Join(w.err, h.reload())
	return w
}

// Add records r as History.Add does.
func (w *HistoryWrite) Add(r Record) (Record, error) {
	r, err := w.h.add(r)
	return r, errors.Join(w.err, err)
}

// End releases the write.
func (w *HistoryWrite) End() {
	w.unlock()
	w.h.writes.Unlock()
}

// reload replaces the records with the store's, which include those other
// writers recorded, and keeps the ones the store failed to persist.
func (h *History) reload() error {
	loaded, err := h.store.Load()
	if err != nil && loaded == nil {
		return err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, r := range h.records {
		if h.unsaved[r.ID] {
			loaded = append(loaded, r)
		}
	}
	h.records = loaded
	for _, r := range loaded {
		h.follow(r.ID)
	}
	h.sort()
	return err
}

// NewHistory returns a history of at most limit records. A store, when given,
// persists the records and supplies those kept before a restart; without one
// the history lives in memory.
func NewHistory(limit int, store HistoryStore) (*History, error) {
	if limit < 1 {
		return nil, fmt.Errorf("the configuration history must keep at least one version, not %d", limit)
	}
	h := &History{limit: limit, store: store, unsaved: map[string]bool{}}
	if store == nil {
		return h, nil
	}
	records, err := store.Load()
	h.records = records
	for _, r := range records {
		h.follow(r.ID)
	}
	h.sort()
	return h, errors.Join(err, h.trim())
}

// Limit is the most records the history keeps.
func (h *History) Limit() int { return h.limit }

// Add records r under a new ID and drops the oldest records past the limit.
// A record is never dropped while it is the newest activation. When the store
// fails to persist r, the history still keeps it and Add reports the error.
// Add is a write of its own; within a HistoryWrite, use its Add.
func (h *History) Add(r Record) (Record, error) {
	w := h.Begin()
	defer w.End()
	return w.Add(r)
}

func (h *History) add(r Record) (Record, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if r.RecordedAt.IsZero() {
		r.RecordedAt = time.Now().UTC()
	}
	r.ID = h.nextID(r.RecordedAt)
	var saveErr error
	if h.store != nil {
		if saved, err := h.store.Save(r); err != nil {
			saveErr = err
			h.unsaved[r.ID] = true
		} else {
			r = saved
		}
	}
	h.follow(r.ID)
	h.records = append(h.records, r)
	h.sort()
	return r, errors.Join(saveErr, h.trim())
}

// nextID is the ID of a record written at at: its time, or the ID after the
// last one when that time was used already or the clock moved back.
func (h *History) nextID(at time.Time) string {
	if id := at.Local().Format(recordIDLayout); id > h.lastID {
		return id
	}
	return nextRecordID(h.lastID)
}

// follow notes that id is taken, so later IDs sort after it.
func (h *History) follow(id string) {
	if recordIDPattern.MatchString(id) && id > h.lastID {
		h.lastID = id
	}
}

// nextRecordID is the ID that follows id within its second.
func nextRecordID(id string) string {
	match := recordIDPattern.FindStringSubmatch(id)
	if match == nil {
		return id + "-001"
	}
	sequence, _ := strconv.Atoi(match[2])
	return fmt.Sprintf("%s-%03d", match[1], sequence+1)
}

// List returns the records, newest first. Their documents are shared and
// must not be modified.
func (h *History) List() []Record {
	h.mu.Lock()
	defer h.mu.Unlock()
	list := make([]Record, len(h.records))
	for i, r := range h.records {
		list[len(h.records)-1-i] = r
	}
	return list
}

// ByID returns the record named id.
func (h *History) ByID(id string) (Record, bool) {
	return h.find(func(r Record) bool { return r.ID == id })
}

// ByVersion returns the record of the activation that took version.
func (h *History) ByVersion(version uint64) (Record, bool) {
	if version == 0 {
		return Record{}, false
	}
	return h.find(func(r Record) bool { return r.Version == version })
}

// ByHash returns the newest record of the document with hash.
func (h *History) ByHash(hash string) (Record, bool) {
	if hash == "" {
		return Record{}, false
	}
	return h.find(func(r Record) bool { return r.Hash == hash })
}

// Latest returns the record of the newest activation.
func (h *History) Latest() (Record, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.latestLocked()
}

func (h *History) latestLocked() (Record, bool) {
	var latest Record
	for _, r := range h.records {
		if r.Version > latest.Version {
			latest = r
		}
	}
	return latest, latest.Version > 0
}

func (h *History) find(match func(Record) bool) (Record, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := len(h.records) - 1; i >= 0; i-- {
		if match(h.records[i]) {
			return h.records[i], true
		}
	}
	return Record{}, false
}

// sort orders records by when they were written; several writers share the
// directory, and their IDs follow different layouts.
func (h *History) sort() {
	sort.SliceStable(h.records, func(i, j int) bool {
		a, b := h.records[i], h.records[j]
		if !a.RecordedAt.Equal(b.RecordedAt) {
			return a.RecordedAt.Before(b.RecordedAt)
		}
		return a.ID < b.ID
	})
}

// trim drops the oldest records past the limit, except the newest activation.
func (h *History) trim() error {
	latest, _ := h.latestLocked()
	var errs []error
	for len(h.records) > h.limit {
		drop := 0
		if h.records[0].ID == latest.ID && latest.Version > 0 {
			drop = 1
		}
		if h.store != nil && !h.unsaved[h.records[drop].ID] {
			if err := h.store.Remove(h.records[drop].ID); err != nil {
				errs = append(errs, err)
			}
		}
		delete(h.unsaved, h.records[drop].ID)
		h.records = append(h.records[:drop], h.records[drop+1:]...)
	}
	return errors.Join(errs...)
}
