# Model selection fixtures

These four small, hand-authored artifacts test the inference format without
training or downloading a model. Each consumes two embedding dimensions followed
by the runtime's fourteen category dimensions. With embedding `[0.25, 0.25]`,
`math` selects `model-a` and `other` selects `model-b`.

The fixtures cover KNN neighbors, KMeans centroids, linear SVM classifiers, and
a single MLP linear layer. They test artifact loading, category feature order,
and candidate matching; they do not measure learned routing quality.
