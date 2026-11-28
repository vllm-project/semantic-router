package store

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// toolEventFile keeps a tool event store in a JSON Lines file, as the file
// replay store keeps its records: each report is appended as it arrives, the
// file is rewritten with the events kept once it holds twice the store's
// limit and when the store closes, and opening it reads the events back,
// dropping those past the TTL. A line cut short by a crash is skipped. The
// file is readable by its owner only, and one process at a time holds it.
type toolEventFile struct {
	path  string
	file  *os.File
	lock  *os.File
	lines int
}

const toolEventRewritePrefix = ".router-tool-events-"

func openToolEventFile(path string, store *ToolEventStore) (*toolEventFile, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	lock, err := lockFile(path + ".lock")
	if err != nil {
		return nil, err
	}
	disk := &toolEventFile{path: path, lock: lock}
	if err = disk.load(store); err != nil {
		_ = lock.Close()
		return nil, err
	}
	return disk, nil
}

// load removes what an interrupted rewrite left, reads the events back into
// the store and opens the file for appending.
func (d *toolEventFile) load(store *ToolEventStore) error {
	dir := filepath.Dir(d.path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), toolEventRewritePrefix) {
			_ = os.Remove(filepath.Join(dir, entry.Name()))
		}
	}
	data, err := os.ReadFile(d.path)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		var event ToolEvent
		if len(line) == 0 {
			continue
		}
		d.lines++
		if json.Unmarshal(line, &event) == nil && event.SessionID != "" && event.ToolCallID != "" {
			store.putLocked(event)
		}
	}
	store.trimLocked(time.Now())
	if d.file, err = os.OpenFile(d.path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600); err != nil {
		return err
	}
	err = d.file.Chmod(0o600)
	if err == nil && len(data) > 0 && data[len(data)-1] != '\n' {
		_, err = d.file.Write([]byte{'\n'})
	}
	if err != nil {
		_ = d.file.Close()
	}
	return err
}

// append writes the events as the store now keeps them.
func (d *toolEventFile) append(store *ToolEventStore, events []ToolEvent) error {
	var lines bytes.Buffer
	for _, event := range events {
		line, err := json.Marshal(event)
		if err != nil {
			return err
		}
		lines.Write(append(line, '\n'))
	}
	if _, err := d.file.Write(lines.Bytes()); err != nil {
		return err
	}
	// The events are written; a failed rewrite leaves the longer file in place
	// and is tried again after another max lines.
	if d.lines += len(events); d.lines >= 2*store.max {
		if err := d.rewrite(store); err != nil {
			d.lines = store.max
			logging.Warnf("router replay could not rewrite %s: %v", d.path, err)
		}
	}
	return nil
}

// rewrite replaces the file with one line for each event kept.
func (d *toolEventFile) rewrite(store *ToolEventStore) error {
	dir := filepath.Dir(d.path)
	tmp, err := os.CreateTemp(dir, toolEventRewritePrefix+"*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	writer := bufio.NewWriter(tmp)
	for _, key := range store.order {
		var line []byte
		if line, err = json.Marshal(store.events[key]); err != nil {
			break
		}
		if _, err = writer.Write(append(line, '\n')); err != nil {
			break
		}
	}
	if err = errors.Join(err, writer.Flush(), tmp.Sync(), tmp.Close()); err != nil {
		return err
	}
	if err = os.Rename(tmp.Name(), d.path); err != nil {
		return err
	}
	if err = syncDir(dir); err != nil {
		return err
	}
	file, err := os.OpenFile(d.path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_ = d.file.Close()
	d.file, d.lines = file, len(store.order)
	return nil
}

// close writes the file whole, without the events past the TTL, and lets it go.
func (d *toolEventFile) close(store *ToolEventStore) error {
	store.trimLocked(time.Now())
	return errors.Join(d.rewrite(store), d.file.Close(), d.lock.Close())
}
