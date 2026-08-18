//go:build !windows

package handlers

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	maxLogSpoolReadBytes      = 4 << 20
	maxRuntimeLogLineBytes    = 64 << 10
	runtimeLogTruncatedMarker = " ... [log record truncated]"
)

var (
	errLogSpoolUnsupported = errors.New("bounded log spool is unsupported")
	logSpoolFileNames      = map[string]string{
		"router":    "router.log",
		"envoy":     "envoy.log",
		"dashboard": "dashboard.log",
	}
)

type logSpoolReader struct {
	rootFD int
}

func openLogSpoolReader(root string) (*logSpoolReader, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, errLogSpoolUnsupported
	}
	if !filepath.IsAbs(root) {
		return nil, errors.New("bounded log spool root must be absolute")
	}
	rootFD, err := unix.Open(
		filepath.Clean(root),
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		if errors.Is(err, unix.ENOENT) {
			return nil, errLogSpoolUnsupported
		}
		return nil, fmt.Errorf("open bounded log spool root: %w", err)
	}
	var info unix.Stat_t
	if err := unix.Fstat(rootFD, &info); err != nil {
		_ = unix.Close(rootFD)
		return nil, fmt.Errorf("inspect bounded log spool root: %w", err)
	}
	if info.Mode&unix.S_IFMT != unix.S_IFDIR {
		_ = unix.Close(rootFD)
		return nil, errors.New("bounded log spool root is not a directory")
	}
	return &logSpoolReader{rootFD: rootFD}, nil
}

func (reader *logSpoolReader) Close() {
	if reader == nil || reader.rootFD < 0 {
		return
	}
	_ = unix.Close(reader.rootFD)
	reader.rootFD = -1
}

func (reader *logSpoolReader) Tail(component string, lineLimit int) ([]string, error) {
	name, ok := logSpoolFileNames[component]
	if !ok || lineLimit < 1 {
		return nil, errors.New("invalid bounded log spool request")
	}
	fd, err := unix.Openat(
		reader.rootFD,
		name,
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	if err != nil {
		if errors.Is(err, unix.ENOENT) {
			return nil, os.ErrNotExist
		}
		return nil, fmt.Errorf("open bounded component log: %w", err)
	}
	file := os.NewFile(uintptr(fd), name)
	if file == nil {
		_ = unix.Close(fd)
		return nil, errors.New("open bounded component log: invalid descriptor")
	}
	defer file.Close()

	var info unix.Stat_t
	if err := unix.Fstat(fd, &info); err != nil {
		return nil, fmt.Errorf("inspect bounded component log: %w", err)
	}
	if info.Mode&unix.S_IFMT != unix.S_IFREG || info.Nlink != 1 {
		return nil, errors.New("bounded component log is not a private regular file")
	}
	if info.Size <= 0 {
		return []string{}, nil
	}

	readBytes := info.Size
	if readBytes > maxLogSpoolReadBytes {
		readBytes = maxLogSpoolReadBytes
	}
	start := info.Size - readBytes
	buffer := make([]byte, int(readBytes))
	count, readErr := file.ReadAt(buffer, start)
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return nil, fmt.Errorf("read bounded component log: %w", readErr)
	}
	buffer = buffer[:count]
	if start > 0 {
		if newline := bytes.IndexByte(buffer, '\n'); newline >= 0 {
			buffer = buffer[newline+1:]
		} else {
			return []string{}, nil
		}
	}

	parts := bytes.Split(buffer, []byte{'\n'})
	if len(parts) > 0 && len(parts[len(parts)-1]) == 0 {
		parts = parts[:len(parts)-1]
	}
	if len(parts) > lineLimit {
		parts = parts[len(parts)-lineLimit:]
	}
	lines := make([]string, 0, len(parts))
	for _, part := range parts {
		if len(part) > maxRuntimeLogLineBytes {
			prefixBytes := maxRuntimeLogLineBytes - len(runtimeLogTruncatedMarker)
			part = append(append([]byte{}, part[:prefixBytes]...), []byte(runtimeLogTruncatedMarker)...)
		}
		line := strings.ToValidUTF8(string(bytes.TrimSuffix(part, []byte{'\r'})), "�")
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines, nil
}
