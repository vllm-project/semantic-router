package milvus

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"
)

type describeCollectionFunc func(context.Context, string) (*entity.Collection, error)

func (f describeCollectionFunc) DescribeCollection(ctx context.Context, name string) (*entity.Collection, error) {
	return f(ctx, name)
}

func TestValidateFloatVectorDimension(t *testing.T) {
	for _, tc := range []struct {
		name    string
		field   *entity.Field
		wantErr string
	}{
		{name: "match", field: entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(384)},
		{name: "mismatch", field: entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(768), wantErr: "dimension 768, expected 384"},
		{name: "no field", wantErr: `missing vector field "embedding"`},
		{name: "wrong type", field: entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeInt64), wantErr: "not a FloatVector"},
		{name: "no dimension", field: entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector), wantErr: "invalid dimension"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := describeCollectionFunc(func(_ context.Context, name string) (*entity.Collection, error) {
				require.Equal(t, "vectors", name)
				schema := entity.NewSchema()
				if tc.field != nil {
					schema.WithField(tc.field)
				}
				return &entity.Collection{Schema: schema}, nil
			})
			err := ValidateFloatVectorDimension(context.Background(), client, "vectors", "embedding", 384)
			if tc.wantErr == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tc.wantErr)
			}
		})
	}
}

func TestValidateFloatVectorDimensionDescribeFailure(t *testing.T) {
	backendErr := errors.New("describe failed")
	client := describeCollectionFunc(func(context.Context, string) (*entity.Collection, error) {
		return nil, backendErr
	})
	require.ErrorIs(t, ValidateFloatVectorDimension(context.Background(), client, "vectors", "embedding", 384), backendErr)
}
