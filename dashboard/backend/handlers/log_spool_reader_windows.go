//go:build windows

package handlers

import "errors"

var errLogSpoolUnsupported = errors.New("bounded log spool is unsupported")

type logSpoolReader struct{}

func openLogSpoolReader(string) (*logSpoolReader, error) {
	return nil, errLogSpoolUnsupported
}

func (*logSpoolReader) Close() {}

func (*logSpoolReader) Tail(string, int) ([]string, error) {
	return nil, errLogSpoolUnsupported
}
