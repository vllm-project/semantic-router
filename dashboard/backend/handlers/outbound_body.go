package handlers

import "io"

func readBoundedOutboundBody(body io.Reader, maxBytes int64) ([]byte, error) {
	if body == nil || maxBytes < 0 {
		return nil, errOutboundRequestFailed
	}
	content, err := io.ReadAll(io.LimitReader(body, maxBytes+1))
	if err != nil {
		return nil, errOutboundRequestFailed
	}
	if int64(len(content)) > maxBytes {
		return nil, errOutboundResponseTooLarge
	}
	return content, nil
}
