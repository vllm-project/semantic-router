package evaluationplane

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const lifecycleCollectionTemporaryPrefix = ".tmp-evaluation-collection-"

// lifecycleCollectionPersistence is the narrow durability seam for the
// immutable plan header and its bounded append-only progress chain.
type lifecycleCollectionPersistence interface {
	WriteHeader(path string, value lifecycleCollectionHeader) error
	AppendProgress(path string, value lifecycleCollectionProgress) error
	Resolve(path, directory string) error
}

type atomicLifecycleCollectionPersistence struct{}

func (atomicLifecycleCollectionPersistence) WriteHeader(
	path string,
	value lifecycleCollectionHeader,
) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return writeLifecycleCollectionHeaderAtomic(path, append(encoded, '\n'))
}

func (atomicLifecycleCollectionPersistence) AppendProgress(
	path string,
	value lifecycleCollectionProgress,
) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	record := encoded
	if len(record) > maxLifecycleCollectionProgressBytes {
		return fmt.Errorf("lifecycle collection progress record exceeds its bound")
	}
	file, err := openBundleFile(path, os.O_WRONLY|os.O_APPEND)
	if err != nil {
		return err
	}
	for len(record) > 0 && err == nil {
		var written int
		written, err = file.Write(record)
		if written == 0 && err == nil {
			err = io.ErrShortWrite
		}
		record = record[written:]
	}
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	return errors.Join(err, closeErr)
}

// Resolve is called only by startup authority or the matching explicit retry.
// It commits a visible namespace mutation and removes an incomplete append tail
// before a transaction can be resumed. A peer opener never calls this method.
func (atomicLifecycleCollectionPersistence) Resolve(path, directory string) error {
	file, err := openBundleFile(path, os.O_RDWR)
	if err != nil {
		return err
	}
	if err = file.Sync(); err == nil {
		err = repairLifecycleCollectionTail(file)
	}
	closeErr := file.Close()
	if err = errors.Join(err, closeErr); err != nil {
		return err
	}
	return syncEvaluationDirectory(directory, "evaluation lifecycle collection transaction retry")
}

func repairLifecycleCollectionTail(file *os.File) error {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	data, err := io.ReadAll(io.LimitReader(file, maxStructuredArtifactBytes+1))
	if err != nil {
		return err
	}
	if int64(len(data)) > maxStructuredArtifactBytes {
		return fmt.Errorf("lifecycle collection transaction exceeds its durable byte bound")
	}
	if len(data) == 0 || data[len(data)-1] == '\n' {
		return nil
	}
	lastComplete := bytes.LastIndexByte(data, '\n')
	if lastComplete < 0 {
		return fmt.Errorf("lifecycle collection transaction header is incomplete")
	}
	if err := file.Truncate(int64(lastComplete + 1)); err != nil {
		return err
	}
	return file.Sync()
}

func requireNoLifecycleCollectionTemps(directory string) error {
	return inspectLifecycleCollectionDirectory(directory, false)
}

func recoverLifecycleCollectionTemps(directory string) error {
	if err := inspectLifecycleCollectionDirectory(directory, true); err != nil {
		return err
	}
	return syncEvaluationDirectory(directory, "recover evaluation lifecycle collection temporary files")
}

func inspectLifecycleCollectionDirectory(directory string, removeTemps bool) error {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("inspect lifecycle collection directory: %w", err)
	}
	for _, entry := range entries {
		if entry.Name() == lifecycleCollectionFileName {
			continue
		}
		if !strings.HasPrefix(entry.Name(), lifecycleCollectionTemporaryPrefix) {
			return fmt.Errorf("%w: lifecycle collection directory contains an unknown entry", ErrInvalid)
		}
		path := filepath.Join(directory, entry.Name())
		info, err := os.Lstat(path)
		if err != nil {
			return fmt.Errorf("inspect lifecycle collection temporary file: %w", err)
		}
		if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
			return fmt.Errorf("%w: lifecycle collection temporary file is invalid", ErrInvalid)
		}
		if !removeTemps {
			return fmt.Errorf("%w: lifecycle collection recovery is required", ErrConflict)
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("remove lifecycle collection temporary file: %w", err)
		}
	}
	return nil
}

func writeLifecycleCollectionHeaderAtomic(path string, encoded []byte) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, lifecycleCollectionTemporaryPrefix+"*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return err
	}
	for len(encoded) > 0 {
		written, err := temporary.Write(encoded)
		if written == 0 && err == nil {
			err = io.ErrShortWrite
		}
		encoded = encoded[written:]
		if err != nil {
			_ = temporary.Close()
			return err
		}
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	return syncEvaluationDirectory(directory, "evaluation lifecycle collection transaction")
}
