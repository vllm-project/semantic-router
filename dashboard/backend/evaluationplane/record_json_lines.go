package evaluationplane

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func scanEvidenceJSONLines(path string, maxBytes int64, maxLineBytes, maxLines int, visit func([]byte, int) error) error {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return fmt.Errorf("open evidence file %s: %w", filepath.Base(path), err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("stat evidence file %s: %w", filepath.Base(path), err)
	}
	if info.Size() < 1 || info.Size() > maxBytes {
		return fmt.Errorf("%w: evidence file %s violates its total-byte limit", ErrInvalid, filepath.Base(path))
	}
	if _, err := file.Seek(-1, io.SeekEnd); err != nil {
		return fmt.Errorf("inspect evidence file %s: %w", filepath.Base(path), err)
	}
	var ending [1]byte
	if _, err := io.ReadFull(file, ending[:]); err != nil || ending[0] != '\n' {
		return fmt.Errorf("%w: evidence file %s must end with a newline", ErrInvalid, filepath.Base(path))
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind evidence file %s: %w", filepath.Base(path), err)
	}
	initialBuffer := maxLineBytes + 1
	if initialBuffer > 64*1024 {
		initialBuffer = 64 * 1024
	}
	scanner := bufio.NewScanner(io.LimitReader(file, maxBytes+1))
	scanner.Buffer(make([]byte, initialBuffer), maxLineBytes+1)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := scanner.Bytes()
		if lineNumber > maxLines {
			return fmt.Errorf("%w: evidence file %s exceeds its line-count limit", ErrInvalid, filepath.Base(path))
		}
		if len(line) == 0 || len(line) > maxLineBytes {
			return fmt.Errorf("%w: evidence file %s line %d violates its line-byte limit", ErrInvalid, filepath.Base(path), lineNumber)
		}
		if err := rejectDuplicateJSONKeys(line); err != nil {
			return fmt.Errorf("%w: evidence file %s line %d is ambiguous: %w", ErrInvalid, filepath.Base(path), lineNumber, err)
		}
		if err := visit(line, lineNumber); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("%w: scan evidence file %s: %w", ErrInvalid, filepath.Base(path), err)
	}
	if lineNumber == 0 {
		return fmt.Errorf("%w: evidence file %s must contain at least one row", ErrInvalid, filepath.Base(path))
	}
	return nil
}

func decodeStrictJSONLine(line []byte, destination any) error {
	if err := rejectDuplicateJSONKeys(line); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
