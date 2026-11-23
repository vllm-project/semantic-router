package milvus

import (
	"context"
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type collectionDescriber interface {
	DescribeCollection(context.Context, string) (*entity.Collection, error)
}

// ValidateFloatVectorDimension checks an existing collection against the store's resolved embedding width.
func ValidateFloatVectorDimension(ctx context.Context, client collectionDescriber, collectionName, fieldName string, expected int) error {
	collection, err := client.DescribeCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("describe collection %q: %w", collectionName, err)
	}
	if collection == nil || collection.Schema == nil {
		return fmt.Errorf("collection %q has no schema", collectionName)
	}
	for _, field := range collection.Schema.Fields {
		if field == nil || field.Name != fieldName {
			continue
		}
		if field.DataType != entity.FieldTypeFloatVector {
			return fmt.Errorf("collection %q field %q is not a FloatVector", collectionName, fieldName)
		}
		rawDimension := field.TypeParams["dim"]
		dimension, err := strconv.Atoi(rawDimension)
		if err != nil || dimension <= 0 {
			return fmt.Errorf("collection %q vector field %q has invalid dimension %q, expected %d", collectionName, fieldName, rawDimension, expected)
		}
		if dimension != expected {
			return fmt.Errorf("collection %q vector field %q has dimension %d, expected %d", collectionName, fieldName, dimension, expected)
		}
		return nil
	}
	return fmt.Errorf("collection %q is missing vector field %q", collectionName, fieldName)
}
