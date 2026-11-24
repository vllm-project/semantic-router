package store

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (f *FileStore) queueRecord(id string) {
	if f.rewritePending {
		return
	}
	if len(f.pending) >= f.limit {
		f.rewritePending = true
	} else if rec, found, _ := f.MemoryStore.Get(context.Background(), id); found {
		f.pending = append(f.pending, rec)
	}
	select {
	case f.wake <- struct{}{}:
	default:
	}
}

func (f *FileStore) writeLoop() {
	defer close(f.done)
	writer := bufio.NewWriterSize(io.Discard, 64<<10)
	var batch []Record
	var err error
	for {
		if err != nil {
			select {
			case <-time.After(time.Second):
			case <-f.stop:
			}
		} else {
			select {
			case <-f.wake:
			case <-f.stop:
			}
		}

		f.mu.Lock()
		closing := f.closed
		rewrite := closing || err != nil || f.rewritePending || f.lines+len(f.pending) >= 2*f.limit
		batch, f.pending = f.pending, batch[:0]
		f.rewritePending = false
		var snapshot []Record
		if rewrite {
			snapshot, _ = f.MemoryStore.List(context.Background())
		}
		f.mu.Unlock()

		hadError := err != nil
		if rewrite {
			err = f.rewrite(writer, snapshot)
		} else {
			err = f.appendBatch(writer, batch)
		}
		clear(batch)
		if closing {
			f.closeErr = errors.Join(err, f.file.Close(), f.lock.Close())
			return
		}
		if err != nil && !hadError {
			logging.Warnf("router replay could not persist %s, retrying from memory: %v", f.path, err)
		}
	}
}

func (f *FileStore) appendBatch(writer *bufio.Writer, batch []Record) error {
	writer.Reset(f.file)
	encoder := json.NewEncoder(writer)
	for i := range batch {
		if err := encoder.Encode(batch[i].JSONWire()); err != nil {
			return err
		}
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	f.lines += len(batch)
	return nil
}

func (f *FileStore) rewrite(writer *bufio.Writer, records []Record) error {
	dir := filepath.Dir(f.path)
	tmp, err := os.CreateTemp(dir, f.rewritePrefix()+"*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	writer.Reset(tmp)
	encoder := json.NewEncoder(writer)
	lines, now := 0, time.Now()
	for i := len(records) - 1; i >= 0; i-- {
		record := &records[i]
		if !recordEnded(record.LifecycleState) || f.expired(*record, now) {
			continue
		}
		if err = encoder.Encode(record.JSONWire()); err != nil {
			break
		}
		lines++
	}
	if err = errors.Join(err, writer.Flush(), tmp.Sync()); err == nil {
		err = os.Rename(tmp.Name(), f.path)
	}
	if err != nil {
		return errors.Join(err, tmp.Close())
	}
	old := f.file
	f.file, f.lines = tmp, lines
	return errors.Join(old.Close(), syncDir(dir))
}
