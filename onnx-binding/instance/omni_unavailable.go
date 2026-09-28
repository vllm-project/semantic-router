//go:build windows || !cgo || (!amd64 && !arm64)

package instance

type OmniModel struct{ *owner }

func LoadOmni(Options) (*OmniModel, error)    { return nil, unavailable }
func (*OmniModel) Clone() (*OmniModel, error) { return nil, unavailable }
func (*OmniModel) EncodeText(string, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}
func (*OmniModel) EncodeImageBytes([]byte, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}
func (*OmniModel) EncodeAudioPCM([]float32, int, int, int) (EmbeddingResult, error) {
	return EmbeddingResult{}, unavailable
}
func (*OmniModel) Windows(string, int) ([]TextWindow, error)  { return nil, unavailable }
func (*OmniModel) RuntimeDescriptor(int, int) (string, error) { return "", unavailable }
