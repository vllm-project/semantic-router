package proxy

import (
	"errors"
	"io"
)

const (
	jaegerHTMLResponseByteLimit            int64 = 2 << 20
	openClawControlConfigResponseByteLimit int64 = 256 << 10
)

var errProxyResponseBodyTooLarge = errors.New("proxy upstream response exceeds the size limit")

func readBoundedProxyResponseBody(body io.ReadCloser, limit int64) ([]byte, error) {
	defer body.Close()

	limited := &io.LimitedReader{R: body, N: limit + 1}
	contents, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(contents)) > limit {
		return nil, errProxyResponseBodyTooLarge
	}
	return contents, nil
}
