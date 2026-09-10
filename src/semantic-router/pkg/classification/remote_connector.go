package classification

import (
	"context"
	"fmt"
	"net/http"
)

func bearerAuthorizer(accessKey string) func(context.Context, *http.Request) error {
	if accessKey == "" {
		return nil
	}
	return func(_ context.Context, request *http.Request) error {
		request.Header.Set("Authorization", fmt.Sprintf("Bearer %s", accessKey))
		return nil
	}
}
