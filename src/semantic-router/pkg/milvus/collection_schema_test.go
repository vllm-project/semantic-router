package milvus

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type fakeCollectionSchemaReader struct {
	collection *entity.Collection
	err        error
}

func (f fakeCollectionSchemaReader) DescribeCollection(context.Context, string) (*entity.Collection, error) {
	return f.collection, f.err
}

func collectionWithVectorDimension(dimension int) *entity.Collection {
	return &entity.Collection{
		Schema: &entity.Schema{
			Fields: []*entity.Field{
				{
					Name:       "embedding",
					DataType:   entity.FieldTypeFloatVector,
					TypeParams: map[string]string{"dim": strconv.Itoa(dimension)},
				},
			},
		},
	}
}

func TestValidateVectorDimension_MatchingDimension(t *testing.T) {
	reader := fakeCollectionSchemaReader{collection: collectionWithVectorDimension(3)}

	if err := ValidateVectorDimension(context.Background(), reader, "memories", "embedding", 3); err != nil {
		t.Fatalf("expected matching dimension to pass, got %v", err)
	}
}

func TestValidateVectorDimension_Mismatch(t *testing.T) {
	reader := fakeCollectionSchemaReader{collection: collectionWithVectorDimension(3)}

	err := ValidateVectorDimension(context.Background(), reader, "memories", "embedding", 4)
	if err == nil {
		t.Fatal("expected dimension mismatch error")
	}
	if !strings.Contains(err.Error(), "stored=3") || !strings.Contains(err.Error(), "expected=4") {
		t.Fatalf("dimension details missing from error: %v", err)
	}
}

func TestValidateVectorDimension_DescribeError(t *testing.T) {
	wantErr := errors.New("milvus unavailable")
	reader := fakeCollectionSchemaReader{err: wantErr}

	err := ValidateVectorDimension(context.Background(), reader, "memories", "embedding", 3)
	if err == nil || !strings.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("expected describe error, got %v", err)
	}
}

func TestValidateVectorDimension_RejectsNonVectorField(t *testing.T) {
	reader := fakeCollectionSchemaReader{
		collection: &entity.Collection{
			Schema: &entity.Schema{
				Fields: []*entity.Field{
					{
						Name:       "embedding",
						DataType:   entity.FieldTypeFloat,
						TypeParams: map[string]string{"dim": "3"},
					},
				},
			},
		},
	}

	if err := ValidateVectorDimension(context.Background(), reader, "memories", "embedding", 3); err == nil {
		t.Fatal("expected non-vector field error")
	}
}
