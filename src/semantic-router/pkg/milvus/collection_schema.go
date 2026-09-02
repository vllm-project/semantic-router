package milvus

import (
	"context"
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CollectionSchemaReader is the subset of the Milvus client needed to inspect
// an existing collection's schema.
type CollectionSchemaReader interface {
	DescribeCollection(
		context.Context,
		string,
	) (*entity.Collection, error)
}

// ValidateVectorDimension verifies that an existing collection contains the
// expected float-vector field and dimension.
func ValidateVectorDimension(
	ctx context.Context,
	reader CollectionSchemaReader,
	collectionName string,
	vectorFieldName string,
	expectedDimension int,
) error {
	if reader == nil {
		return fmt.Errorf("collection schema reader is required")
	}
	if expectedDimension <= 0 {
		return fmt.Errorf("expected vector dimension must be positive: %d", expectedDimension)
	}

	collection, err := reader.DescribeCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to describe collection %s: %w", collectionName, err)
	}
	if collection == nil || collection.Schema == nil {
		return fmt.Errorf("collection %s has no schema", collectionName)
	}

	for _, field := range collection.Schema.Fields {
		if field == nil || field.Name != vectorFieldName {
			continue
		}
		if field.DataType != entity.FieldTypeFloatVector {
			return fmt.Errorf(
				"collection %s field %s has type %v, expected float vector",
				collectionName,
				vectorFieldName,
				field.DataType,
			)
		}

		rawDimension, ok := field.TypeParams["dim"]
		if !ok {
			return fmt.Errorf(
				"collection %s vector field %s has no dimension",
				collectionName,
				vectorFieldName,
			)
		}

		storedDimension, err := strconv.Atoi(rawDimension)
		if err != nil {
			return fmt.Errorf(
				"collection %s vector field %s has invalid dimension %q",
				collectionName,
				vectorFieldName,
				rawDimension,
			)
		}

		if storedDimension != expectedDimension {
			return fmt.Errorf(
				"collection %s vector dimension mismatch: stored=%d expected=%d",
				collectionName,
				storedDimension,
				expectedDimension,
			)
		}

		return nil
	}

	return fmt.Errorf(
		"collection %s vector field %s not found",
		collectionName,
		vectorFieldName,
	)
}
