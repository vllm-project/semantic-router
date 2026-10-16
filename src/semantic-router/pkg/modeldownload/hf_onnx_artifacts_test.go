package modeldownload

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func onnxFixtureField(number uint64, value []byte) []byte {
	data := binary.AppendUvarint(nil, number<<3|2)
	data = binary.AppendUvarint(data, uint64(len(value)))
	return append(data, value...)
}

func onnxFixtureTensor(location string) []byte {
	entry := append(onnxFixtureField(1, []byte("location")), onnxFixtureField(2, []byte(location))...)
	tensor := onnxFixtureField(13, entry)
	tensor = binary.AppendUvarint(tensor, 14<<3)
	return binary.AppendUvarint(tensor, 1)
}

func TestHFONNXExternalDataIncludesConstantSubgraphSparseAndFunction(t *testing.T) {
	cases := map[string][]byte{
		"initializer": onnxFixtureField(7, onnxFixtureField(5, onnxFixtureTensor("data/custom.payload"))),
		"constant":    onnxFixtureField(7, onnxFixtureField(1, onnxFixtureField(5, onnxFixtureField(5, onnxFixtureTensor("data/custom.payload"))))),
		"subgraph":    onnxFixtureField(7, onnxFixtureField(1, onnxFixtureField(5, onnxFixtureField(6, onnxFixtureField(5, onnxFixtureTensor("data/custom.payload")))))),
		"sparse":      onnxFixtureField(7, onnxFixtureField(15, onnxFixtureField(1, onnxFixtureTensor("data/custom.payload")))),
		"function":    onnxFixtureField(25, onnxFixtureField(7, onnxFixtureField(5, onnxFixtureField(5, onnxFixtureTensor("data/custom.payload"))))),
	}
	for name, graph := range cases {
		t.Run(name, func(t *testing.T) {
			spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			writeHFRevisionArtifact(t, spec, "onnx/model.onnx", string(graph), false)
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("missing external data accepted: %v %v", current, err)
			}
			writeHFRevisionArtifact(t, spec, "onnx/data/custom.payload", "tensor weights", true)
			if current, err := hasCurrentModelRevision(spec); err != nil || !current {
				t.Fatalf("valid external data rejected: %v %v", current, err)
			}
			if err := os.Remove(filepath.Join(spec.LocalPath, "onnx/data/custom.payload")); err != nil {
				t.Fatal(err)
			}
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("metadata alone accepted missing external bytes: %v %v", current, err)
			}
		})
	}
}

func TestHFONNXMalformedOrEscapingExternalDataRejected(t *testing.T) {
	for name, graph := range map[string][]byte{
		"escape":           onnxFixtureField(7, onnxFixtureField(5, onnxFixtureTensor("../outside"))),
		"truncated":        {0x3a, 0x20, 0x01},
		"missing-location": onnxFixtureField(7, onnxFixtureField(5, []byte{14 << 3, 1})),
	} {
		t.Run(name, func(t *testing.T) {
			spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			writeHFRevisionArtifact(t, spec, "model.onnx", string(graph), false)
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("invalid ONNX accepted: %v %v", current, err)
			}
		})
	}
}
