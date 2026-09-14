//go:build windows || !cgo || (!amd64 && !arm64)

package instance

var unavailable = &Error{Kind: "capability", Message: "ORT native instances require cgo on a supported platform"}

type (
	owner              struct{}
	SequenceClassifier struct{ *owner }
	TokenClassifier    struct{ *owner }
	EmbeddingModel     struct{ *owner }
	MultiModalModel    struct{ *owner }
)

func (*owner) Close() error                                       { return nil }
func (*owner) Info() (Info, error)                                { return Info{}, unavailable }
func (*owner) FinishProfiling() ([]string, error)                 { return nil, unavailable }
func LoadSequenceClassifier(Options) (*SequenceClassifier, error) { return nil, unavailable }
func (*SequenceClassifier) Clone() (*SequenceClassifier, error)   { return nil, unavailable }
func LoadTokenClassifier(Options) (*TokenClassifier, error)       { return nil, unavailable }
func (*TokenClassifier) Clone() (*TokenClassifier, error)         { return nil, unavailable }
func LoadEmbeddingModel(Options) (*EmbeddingModel, error)         { return nil, unavailable }
func (*EmbeddingModel) Clone() (*EmbeddingModel, error)           { return nil, unavailable }
func LoadMultiModal(Options) (*MultiModalModel, error)            { return nil, unavailable }
func (*MultiModalModel) Clone() (*MultiModalModel, error)         { return nil, unavailable }
func (*SequenceClassifier) Classify(string) (Distribution, error) { return Distribution{}, unavailable }
func (*TokenClassifier) Detect(string) (TokenSpans, error)        { return TokenSpans{}, unavailable }
func (*EmbeddingModel) Encode(string, int, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}

func (*MultiModalModel) EncodeText(string, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}

func (*MultiModalModel) EncodeImage([]float32, int, int, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}

func (*MultiModalModel) EncodeImageBytes([]byte, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}

func (*MultiModalModel) EncodeAudio([]float32, int, int, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}

func (*EmbeddingModel) Windows(string, int) ([]TextWindow, error)  { return nil, unavailable }
func (*MultiModalModel) Windows(string, int) ([]TextWindow, error) { return nil, unavailable }
