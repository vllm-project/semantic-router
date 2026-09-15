package evaluationplane

import (
	"fmt"
	"os"
)

// runEventPersistence is the narrow append durability seam for a run's
// immutable control-event log. The log file already exists in the atomically
// published run bundle, so file sync is the commit boundary for appended data.
type runEventPersistence interface {
	Append(path string, encoded []byte) error
	Sync(path, description string) error
}

type atomicRunEventPersistence struct{}

func (atomicRunEventPersistence) Append(path string, encoded []byte) error {
	file, err := openBundleFile(path, os.O_WRONLY|os.O_APPEND)
	if err != nil {
		return fmt.Errorf("open evaluation event log: %w", err)
	}
	_, writeErr := file.Write(encoded)
	if writeErr == nil {
		writeErr = file.Sync()
	}
	closeErr := file.Close()
	if writeErr != nil {
		return fmt.Errorf("append evaluation event: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close evaluation event log: %w", closeErr)
	}
	return nil
}

func (atomicRunEventPersistence) Sync(path, description string) error {
	file, err := openBundleFile(path, os.O_RDWR)
	if err != nil {
		return fmt.Errorf("open %s: %w", description, err)
	}
	syncErr := file.Sync()
	closeErr := file.Close()
	if syncErr != nil {
		return fmt.Errorf("sync %s: %w", description, syncErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %s: %w", description, closeErr)
	}
	return nil
}
