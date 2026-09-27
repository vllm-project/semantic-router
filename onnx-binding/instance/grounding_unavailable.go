//go:build windows || !cgo || (!amd64 && !arm64)

package instance

type GroundedClassifier struct{ *owner }

func LoadGroundedClassifier(Options) (*GroundedClassifier, error) { return nil, unavailable }
func (*GroundedClassifier) Detect(string, string, string) (TokenSpans, error) {
	return TokenSpans{}, unavailable
}
