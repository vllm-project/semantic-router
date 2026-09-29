# Offline Omni processor references

These compact fixtures validate published preprocessing without model weights.
They were generated with `tools/models/vela_omni/dsp_reference.py`, PyTorch and
torchaudio 2.8.0, Transformers 4.57.6, and the immutable source/processor revisions
in `tools/models/vela_omni/sources.json`.

Each variant keeps the original short float32 PCM inputs, a small PNG, RGB,
grayscale and CMYK JPEGs, three real routing JPEG fixtures, exact
processor filters, and 96 numeric probes from each full reference array. Index
file references to omitted intermediate arrays describe the complete generated
fixture; normal unit tests use the retained probes. The ignored
`full_reference_dsp` test accepts the complete reference fixture through
`VELA_OMNI_DSP_DIR` and compares every element, including spectrogram isolation
from exactly the same resampled waveform.

The native image path matches Pillow pixels exactly, including JPEG chroma
upsampling and CMYK conversion through statically linked libjpeg-turbo. The
complete reference additionally verifies every decoded RGB byte before resize.
Audio comparisons allow
float32 convolution accumulation differences from torchaudio (observed below
3e-8 in waveforms); CLAP's logarithm amplifies these near silent frequency bins.
Using the same resampled waveform yields identical CLAP spectrograms. Real-model
source parity is a separate explicit test and covers final embedding vectors.
