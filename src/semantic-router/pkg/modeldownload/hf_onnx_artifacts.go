package modeldownload

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Read the ONNX protobuf envelope without materializing inline tensor payloads.
// This covers initializers, Constant attributes, nested graphs, sparse tensors
// and local function bodies. External files may have arbitrary names/extensions.
func hfONNXExternalFiles(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var locations []string
	err = walkHFONNXMessage(file, "model", 0, &locations)
	return locations, err
}

func walkHFONNXMessage(input io.Reader, kind string, depth int, locations *[]string) error {
	if depth > 64 {
		return fmt.Errorf("ONNX graph nesting exceeds validation depth")
	}
	reader := bufio.NewReader(input)
	var location string
	external := false
	for {
		tag, err := binary.ReadUvarint(reader)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("invalid ONNX protobuf field: %w", err)
		}
		if tag>>3 == 0 {
			return fmt.Errorf("invalid ONNX protobuf field number")
		}
		number, wire := tag>>3, tag&7
		switch wire {
		case 0:
			value, err := binary.ReadUvarint(reader)
			if err != nil {
				return err
			}
			if kind == "tensor" && number == 14 && value == 1 {
				external = true
			}
		case 1, 5:
			size := int64(8)
			if wire == 5 {
				size = 4
			}
			if _, err := io.CopyN(io.Discard, reader, size); err != nil {
				return err
			}
		case 2:
			size, err := binary.ReadUvarint(reader)
			if err != nil || size > math.MaxInt64 {
				return fmt.Errorf("invalid ONNX field size")
			}
			field := &io.LimitedReader{R: reader, N: int64(size)}
			if kind == "tensor" && number == 13 {
				key, value, err := hfONNXExternalEntry(field)
				if err != nil {
					return err
				}
				if key == "location" {
					location = value
				}
			} else if child := hfONNXChildMessage(kind, number); child != "" {
				if err := walkHFONNXMessage(field, child, depth+1, locations); err != nil {
					return err
				}
			}
			if _, err := io.Copy(io.Discard, field); err != nil {
				return err
			}
			if field.N != 0 {
				return io.ErrUnexpectedEOF
			}
		default:
			return fmt.Errorf("unsupported ONNX protobuf wire type %d", wire)
		}
	}
	if external && location == "" {
		return fmt.Errorf("external tensor is missing its data location")
	}
	if location != "" {
		*locations = append(*locations, location)
	}
	return nil
}

func hfONNXChildMessage(kind string, field uint64) string {
	switch kind {
	case "model":
		if field == 7 {
			return "graph"
		}
		if field == 25 {
			return "function"
		}
	case "graph":
		if field == 1 {
			return "node"
		}
		if field == 5 {
			return "tensor"
		}
		if field == 15 {
			return "sparse"
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
	case "function":
		if field == 7 {
			return "node"
		}
		if field == 11 {
			return "attribute"
		}
	}
	return ""
}

func hfONNXExternalEntry(input *io.LimitedReader) (string, string, error) {
	// Keys and file paths are small; reject a malformed metadata allocation.
	if input.N > 1<<20 {
		return "", "", fmt.Errorf("ONNX external data entry is too large")
	}
	reader := bufio.NewReader(input)
	values := map[uint64]string{}
	for {
		tag, err := binary.ReadUvarint(reader)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil || tag&7 != 2 || (tag>>3 != 1 && tag>>3 != 2) {
			return "", "", fmt.Errorf("invalid ONNX external data entry")
		}
		size, err := binary.ReadUvarint(reader)
		if err != nil || size > 1<<20 {
			return "", "", fmt.Errorf("invalid ONNX external data string")
		}
		data := make([]byte, int(size))
		if _, err := io.ReadFull(reader, data); err != nil {
			return "", "", err
		}
		values[tag>>3] = string(data)
	}
	return values[1], values[2], nil
}
