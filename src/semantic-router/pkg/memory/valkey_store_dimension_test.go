package memory

import "testing"

func TestValkeyIndexVectorDimensionParsesRESPMetadata(t *testing.T) {
	raw := []interface{}{
		"index_definition",
		[]interface{}{"key_type", "HASH"},
		"attributes",
		[]interface{}{
			[]interface{}{"identifier", "embedding", "attribute", "embedding", "type", "VECTOR", "dim", int64(768)},
		},
	}

	dimension, err := valkeyIndexVectorDimension(raw)
	if err != nil || dimension != 768 {
		t.Fatalf("dimension = %d, err=%v; want 768", dimension, err)
	}
}

func TestValkeyIndexVectorDimensionParsesMapMetadata(t *testing.T) {
	raw := map[string]interface{}{
		"attributes": []interface{}{
			map[string]interface{}{"type": "VECTOR", "dim": "256"},
		},
	}

	dimension, err := valkeyIndexVectorDimension(raw)
	if err != nil || dimension != 256 {
		t.Fatalf("dimension = %d, err=%v; want 256", dimension, err)
	}
}

func TestValkeyIndexVectorDimensionParsesRESPBytes(t *testing.T) {
	raw := []interface{}{
		[]byte("attributes"), []interface{}{
			[]interface{}{[]byte("type"), []byte("VECTOR"), []byte("dim"), []byte("512")},
		},
	}

	dimension, err := valkeyIndexVectorDimension(raw)
	if err != nil || dimension != 512 {
		t.Fatalf("dimension = %d, err=%v; want 512", dimension, err)
	}
}

func TestValkeyIndexVectorDimensionSelectsMemoryField(t *testing.T) {
	raw := []interface{}{
		"attributes", []interface{}{
			[]interface{}{"identifier", "other_vector", "type", "VECTOR", "dim", int64(384)},
			[]interface{}{"identifier", "embedding", "type", "VECTOR", "dim", int64(768)},
		},
	}

	dimension, err := valkeyIndexVectorDimension(raw)
	if err != nil || dimension != 768 {
		t.Fatalf("dimension = %d, err=%v; want embedding dimension 768", dimension, err)
	}
}

func TestValkeyIndexVectorDimensionRejectsMissingMetadata(t *testing.T) {
	if _, err := valkeyIndexVectorDimension([]interface{}{"attributes", []interface{}{}}); err == nil {
		t.Fatal("missing vector dimension was accepted")
	}
}
