package maskingregistry

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// Each commit is a frame: the JSON record's length and a CRC-32 of that length,
// the record, and its SHA-256. A frame cut short at the end of the log is an
// append that never completed and is discarded. A frame whose length or
// record does not check is corruption, which stops recovery, since cutting
// the log there would drop allocations that were acknowledged.
// The separate identity anchor distinguishes first start from a missing log.
type File struct {
	mu        sync.Mutex
	path      string
	log, lock *os.File
	header    string
	size      int64
	failed    error
	groups    map[string]*group
	templates map[string]string
}
type group struct {
	next      int64
	exhausted bool
	values    map[string]int64
	indices   map[int64]bool
}
type record struct {
	Header   string  `json:"header,omitempty"`
	Scope    string  `json:"scope,omitempty"`
	Type     string  `json:"type,omitempty"`
	Digest   string  `json:"digest,omitempty"`
	Index    int64   `json:"index,omitempty"`
	Reserved []int64 `json:"reserved,omitempty"`
	Template string  `json:"template,omitempty"`
}

func syncDir(path string) error {
	d, e := os.Open(filepath.Dir(path))
	if e != nil {
		return e
	}
	defer d.Close()
	return d.Sync()
}

func privateCreate(path string, data []byte) error {
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if e != nil {
		return e
	}
	_, e = f.Write(data)
	if e == nil {
		e = f.Sync()
	}
	c := f.Close()
	if e == nil {
		e = c
	}
	if e == nil {
		e = syncDir(path)
	}
	return e
}

func OpenFile(path, namespace, keyIdentity string) (_ *File, err error) {
	if path == "" {
		return nil, errors.New("masking registry: file path is required")
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	f := &File{path: path, header: string(encode("vsr-masking-log-v1", namespace, keyIdentity)), groups: map[string]*group{}, templates: map[string]string{}}
	f.lock, err = os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	// A Router that is restarting can still hold the lock while it drains, so
	// wait for it a while, never indefinitely. In-process users share Open.
	if err = lockWithin(f.lock, lockWait); err != nil {
		return nil, err
	}
	anchor := []byte(hex.EncodeToString([]byte(f.header)))
	existing, e := os.ReadFile(path + ".identity")
	if errors.Is(e, os.ErrNotExist) {
		if _, e = os.Lstat(path); !errors.Is(e, os.ErrNotExist) {
			return nil, errors.New("masking registry: log exists without identity anchor")
		}
		if err = privateCreate(path+".identity", anchor); err != nil {
			return nil, err
		}
		if err = privateCreate(path, nil); err != nil {
			return nil, err
		}
		f.log, err = os.OpenFile(path, os.O_RDWR, 0o600)
		if err != nil {
			return nil, err
		}
		if err = f.commit(record{Header: f.header}); err != nil {
			return nil, err
		}
		return f, nil
	}
	if e != nil || !bytes.Equal(existing, anchor) {
		return nil, errors.New("masking registry: namespace/key identity mismatch")
	}
	f.log, err = os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		return nil, errors.New("masking registry: established log is missing or inaccessible")
	}
	if err = f.recover(); err != nil {
		return nil, err
	}
	return f, nil
}

// maxRecordBytes bounds one log record, such as the reservations of one
// request.
const maxRecordBytes = 16 << 20

// lockWait bounds how long OpenFile waits for another process's lock.
var lockWait = 30 * time.Second

func lockWithin(lock *os.File, wait time.Duration) error {
	deadline := time.Now().Add(wait)
	for {
		err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) || time.Now().After(deadline) {
			return fmt.Errorf("masking registry: %s is locked by another process", lock.Name())
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func (f *File) Close() error {
	if f.log != nil {
		_ = f.log.Close()
	}
	if f.lock != nil {
		_ = syscall.Flock(int(f.lock.Fd()), syscall.LOCK_UN)
		return f.lock.Close()
	}
	return nil
}

func (f *File) recover() error {
	at := int64(0)
	for {
		var head [8]byte
		n, e := io.ReadFull(f.log, head[:])
		if errors.Is(e, io.EOF) && n == 0 {
			break
		}
		if e != nil {
			if errors.Is(e, io.ErrUnexpectedEOF) {
				break
			}
			return e
		}
		length := binary.BigEndian.Uint32(head[:4])
		if crc32.ChecksumIEEE(head[:4]) != binary.BigEndian.Uint32(head[4:]) || length > maxRecordBytes {
			return errors.New("masking registry: corrupt record length")
		}
		data := make([]byte, int(length)+32)
		if _, e = io.ReadFull(f.log, data); e != nil {
			if errors.Is(e, io.EOF) || errors.Is(e, io.ErrUnexpectedEOF) {
				break
			}
			return e
		}
		sum := sha256.Sum256(data[:length])
		if !bytes.Equal(sum[:], data[length:]) {
			return errors.New("masking registry: corrupt record")
		}
		var r record
		if e = json.Unmarshal(data[:length], &r); e != nil {
			return e
		}
		if at == 0 {
			if r.Header != f.header {
				return errors.New("masking registry: log header mismatch")
			}
		} else {
			if r.Header != "" {
				return errors.New("masking registry: unexpected header")
			}
			f.apply(r)
		}
		at += int64(len(head) + len(data))
	}
	if at == 0 {
		return errors.New("masking registry: established log has no header")
	}
	if e := f.log.Truncate(at); e != nil {
		return e
	}
	if e := f.log.Sync(); e != nil {
		return e
	}
	_, e := f.log.Seek(at, io.SeekStart)
	f.size = at
	return e
}

func (f *File) healthy() error {
	if f.failed != nil {
		return f.failed
	}
	current, e := os.Stat(f.path)
	if e != nil {
		return errors.New("masking registry: log disappeared")
	}
	open, e := f.log.Stat()
	if e != nil || !os.SameFile(current, open) || current.Size() != f.size {
		return errors.New("masking registry: log replaced or truncated")
	}
	anchor, e := os.ReadFile(f.path + ".identity")
	if e != nil || string(anchor) != hex.EncodeToString([]byte(f.header)) {
		return errors.New("masking registry: identity anchor changed")
	}
	return nil
}

func (f *File) commit(r record) error {
	b, e := json.Marshal(r)
	if e != nil {
		return e
	}
	if len(b) > maxRecordBytes {
		// Refused before anything is written, so the log stays usable.
		return errors.New("masking registry: record too large")
	}
	sum := sha256.Sum256(b)
	frame := binary.BigEndian.AppendUint32(nil, uint32(len(b))) //nolint:gosec // bounded by maxRecordBytes
	frame = binary.BigEndian.AppendUint32(frame, crc32.ChecksumIEEE(frame))
	frame = append(frame, b...)
	frame = append(frame, sum[:]...)
	n, e := f.log.Write(frame)
	if e == nil && n != len(frame) {
		e = io.ErrShortWrite
	}
	if e == nil {
		e = f.log.Sync()
	}
	if e != nil {
		f.failed = errors.New("masking registry: commit failed; reopen required")
		return f.failed
	}
	f.size += int64(len(frame))
	f.apply(r)
	return nil
}

func (f *File) group(scope, typ string) *group {
	key := string(encode(scope, typ))
	g := f.groups[key]
	if g == nil {
		g = &group{values: map[string]int64{}, indices: map[int64]bool{}}
		f.groups[key] = g
	}
	return g
}

func (f *File) apply(r record) {
	if r.Header != "" {
		return
	}
	if r.Template != "" {
		f.templates[r.Template] = r.Type
		return
	}
	g := f.group(r.Scope, r.Type)
	for _, i := range r.Reserved {
		if _, ok := g.indices[i]; !ok {
			g.indices[i] = false
		}
	}
	if r.Digest != "" {
		g.values[r.Digest] = r.Index
		g.indices[r.Index] = true
		if r.Index == math.MaxInt64 {
			g.exhausted = true
		} else {
			g.next = r.Index + 1
		}
	}
}

func (f *File) Claim(typ, template string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if e := f.healthy(); e != nil {
		return e
	}
	if owner, ok := f.templates[template]; ok {
		if owner != typ {
			return fmt.Errorf("masking registry: template belongs to another canonical type")
		}
		return nil
	}
	return f.commit(record{Type: typ, Template: template})
}

func (f *File) Prepare(scope, typ string, indices []int64) (map[int64]bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if e := f.healthy(); e != nil {
		return nil, e
	}
	g := f.group(scope, typ)
	issued := map[int64]bool{}
	var reserved []int64
	for _, i := range indices {
		if i < 0 {
			return nil, errors.New("masking registry: negative index")
		}
		v, ok := g.indices[i]
		issued[i] = v
		if !ok {
			reserved = append(reserved, i)
		}
	}
	if len(reserved) > 0 {
		if e := f.commit(record{Scope: scope, Type: typ, Reserved: reserved}); e != nil {
			return nil, e
		}
	}
	return issued, nil
}

func (f *File) Allocate(scope, typ, digest string) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if e := f.healthy(); e != nil {
		return 0, e
	}
	g := f.group(scope, typ)
	if i, ok := g.values[digest]; ok {
		return i, nil
	}
	if g.exhausted {
		return 0, errors.New("masking registry: index exhausted")
	}
	i := g.next
	for {
		if _, ok := g.indices[i]; !ok {
			break
		}
		if i == math.MaxInt64 {
			return 0, errors.New("masking registry: index exhausted")
		}
		i++
	}
	if e := f.commit(record{Scope: scope, Type: typ, Digest: digest, Index: i}); e != nil {
		return 0, e
	}
	return i, nil
}
