package candle_binding

// Backbone owns a ModernBERT/mmBERT encoder independently of all task heads.
// Its artifact does not need classifier weights or labels. Binding a head reads
// only that head's weights, tokenizer and labels and shares this actual encoder.
type Backbone struct{ *instance }

func LoadBackbone(options InstanceOptions) (*Backbone, error) {
	i, err := loadInstance(options, "backbone")
	if err != nil {
		return nil, err
	}
	return &Backbone{i}, nil
}

func (m *Backbone) Clone() (*Backbone, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &Backbone{i}, nil
}

func (m *Backbone) BindSequenceHead(path string) (*SequenceClassifier, error) {
	i, err := m.instance.bindHead(path, "sequence")
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{i}, nil
}

func (m *Backbone) BindTokenHead(path string) (*TokenClassifier, error) {
	i, err := m.instance.bindHead(path, "token")
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{i}, nil
}
