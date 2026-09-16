package modeldownload

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// ONNX external_data locations are declared in TensorProto, including tensors
// in node attributes and nested graphs. Read those paths without reading large
// raw tensor payloads into memory. Field numbers follow onnx/onnx.proto.
func onnxExternalFiles(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	var locations []string
	err = walkONNXFields(file, 0, info.Size(), "model", 0, &locations)
	return uniqueStrings(locations), err
}

func walkONNXFields(file *os.File, offset, size int64, message string, depth int, locations *[]string) error {
	if depth > 64 {
		return fmt.Errorf("ONNX nested graph depth exceeds 64")
	}
	section := io.NewSectionReader(file, offset, size)
	reader := bufio.NewReader(section)
	position := func() int64 { n, _ := section.Seek(0, io.SeekCurrent); return n - int64(reader.Buffered()) }
	var key, value string
	for position() < size {
		tag, err := binary.ReadUvarint(reader)
		if err != nil {
			return err
		}
		fieldNumber := tag >> 3
		if fieldNumber > (1<<29)-1 {
			return fmt.Errorf("invalid ONNX protobuf field number")
		}
		field, wire := fieldNumber, tag&7
		if field == 0 {
			return fmt.Errorf("invalid ONNX protobuf field")
		}
		switch wire {
		case 0:
			if _, err := binary.ReadUvarint(reader); err != nil {
				return err
			}
		case 1, 5:
			length := int64(8)
			if wire == 5 {
				length = 4
			}
			next := position() + length
			if next > size {
				return io.ErrUnexpectedEOF
			}
			if _, err := section.Seek(next, io.SeekStart); err != nil {
				return err
			}
			reader.Reset(section)
		case 2:
			length, err := binary.ReadUvarint(reader)
			if err != nil {
				return err
			}
			start := position()
			if length > math.MaxInt64 || int64(length) > size-start {
				return io.ErrUnexpectedEOF
			}
			child := onnxChildMessage(message, field)
			if child != "" {
				if err := walkONNXFields(file, offset+start, int64(length), child, depth+1, locations); err != nil {
					return err
				}
			}
			if message == "entry" && (field == 1 || field == 2) {
				if length > 4096 {
					return fmt.Errorf("ONNX external tensor location is too long")
				}
				bytes := make([]byte, int(length))
				if _, err := file.ReadAt(bytes, offset+start); err != nil {
					return err
				}
				if field == 1 {
					key = string(bytes)
				} else {
					value = string(bytes)
				}
			}
			if _, err := section.Seek(start+int64(length), io.SeekStart); err != nil {
				return err
			}
			reader.Reset(section)
		default:
			return fmt.Errorf("unsupported ONNX protobuf wire type %d", wire)
		}
	}
	if message == "entry" && key == "location" && value != "" {
		*locations = append(*locations, value)
	}
	return nil
}

func onnxChildMessage(message string, field uint64) string {
	switch message {
	case "model":
		if field == 7 {
			return "graph"
		}
		if field == 25 {
			return "function"
		}
	case "graph":
		switch field {
		case 1:
			return "node"
		case 5:
			return "tensor"
		case 15:
			return "sparse"
		}
	case "function":
		if field == 7 {
			return "node"
		}
		if field == 11 {
			return "attribute"
		}
	case "node":
		if field == 5 {
			return "attribute"
		}
	case "attribute":
		switch field {
		case 5, 10:
			return "tensor"
		case 6, 11:
			return "graph"
		case 22, 23:
			return "sparse"
		}
	case "sparse":
		if field == 1 || field == 2 {
			return "tensor"
		}
	case "tensor":
		if field == 13 {
			return "entry"
		}
	}
	return ""
}

func onnxDependenciesPresent(path string) (bool, error) {
	locations, err := onnxExternalFiles(path)
	if err != nil {
		return false, err
	}
	for _, location := range locations {
		if filepath.IsAbs(location) || !filepath.IsLocal(location) {
			return false, fmt.Errorf("ONNX external tensor location must stay relative to its graph")
		}
		info, err := os.Stat(filepath.Join(filepath.Dir(path), location))
		if os.IsNotExist(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if info.IsDir() {
			return false, nil
		}
	}
	return true, nil
}
