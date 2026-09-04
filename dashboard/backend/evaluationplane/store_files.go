package evaluationplane

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
)

func writeJSONAtomic(path string, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode evaluation bundle: %w", err)
	}
	encoded = append(encoded, '\n')
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".tmp-evaluation-*")
	if err != nil {
		return fmt.Errorf("stage evaluation bundle: %w", err)
	}
	tempName := temp.Name()
	defer func() { _ = os.Remove(tempName) }()
	if err := temp.Chmod(0o600); err != nil {
		_ = temp.Close()
		return fmt.Errorf("protect staged evaluation bundle: %w", err)
	}
	if _, err := temp.Write(encoded); err != nil {
		_ = temp.Close()
		return fmt.Errorf("write staged evaluation bundle: %w", err)
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return fmt.Errorf("sync staged evaluation bundle: %w", err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close staged evaluation bundle: %w", err)
	}
	if err := os.Rename(tempName, path); err != nil {
		return fmt.Errorf("publish evaluation bundle: %w", err)
	}
	return nil
}

func readJSON(path string, destination any) error {
	data, err := readBundleFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: evaluation bundle file", ErrNotFound)
		}
		return fmt.Errorf("read evaluation bundle: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode evaluation bundle: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return err
	}
	return nil
}

func openBundleFile(path string, flags int) (*os.File, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("evaluation bundle file is not a regular file")
	}
	if before.Mode().Perm() != 0o600 {
		return nil, fmt.Errorf("evaluation bundle file must have mode 0600")
	}
	file, err := os.OpenFile(path, flags, 0)
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		_ = file.Close()
		return nil, fmt.Errorf("evaluation bundle file changed while opening")
	}
	return file, nil
}

func readBundleFile(path string) ([]byte, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	return io.ReadAll(file)
}

func requirePrivateDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("evaluation store path is not a directory")
	}
	if info.Mode().Perm() != 0o700 {
		return fmt.Errorf("evaluation store directory %s must have mode 0700", filepath.Base(path))
	}
	return nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("evaluation bundle contains trailing JSON")
		}
		return fmt.Errorf("decode evaluation bundle: %w", err)
	}
	return nil
}

func lastEventSequence(path string) (uint64, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return 0, fmt.Errorf("open evaluation event log: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return 0, fmt.Errorf("stat evaluation event log: %w", err)
	}
	if info.Size() > maxEventLogBytes {
		return 0, fmt.Errorf("%w: evaluation event log exceeds its per-run byte limit", ErrInvalid)
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	var last uint64
	var count uint64
	for scanner.Scan() {
		count++
		if count > maxEventsPerRun {
			return 0, fmt.Errorf("%w: evaluation event log exceeds its per-run event limit", ErrInvalid)
		}
		var event Event
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return 0, fmt.Errorf("decode evaluation event: %w", err)
		}
		sequence, err := strconv.ParseUint(event.ID, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("decode evaluation event id: %w", err)
		}
		if sequence > last {
			last = sequence
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("scan evaluation event log: %w", err)
	}
	return last, nil
}
