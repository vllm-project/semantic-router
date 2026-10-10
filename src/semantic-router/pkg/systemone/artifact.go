package systemone

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

func readArtifact(path, digest string, result any) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open inference artifact: %w", err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, (4<<20)+1))
	if err != nil || len(data) > 4<<20 {
		return errors.New("inference artifact exceeds limit or cannot be read")
	}
	sum := sha256.Sum256(data)
	if !strings.EqualFold(hex.EncodeToString(sum[:]), digest) {
		return errors.New("inference artifact digest mismatch")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(result); err != nil {
		return fmt.Errorf("invalid inference artifact: %w", err)
	}
	if decoder.Decode(new(any)) != io.EOF {
		return errors.New("inference artifact has trailing data")
	}
	return nil
}
