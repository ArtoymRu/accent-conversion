NOTE — Why 'first_try' results are inflated (~99%)
===================================================
The random split assigns utterances from the SAME speaker to both train and
test sets.  HuBERT features (especially lower layers) strongly encode speaker
identity (timbre, pitch, vocal-tract shape) — effectively a voice fingerprint.
With 4 speakers per accent, classifiers learn to recognise voices, not accents.

The 'speaker_independent' evaluation holds out one entire speaker per accent
(last alphabetically: ZHAA, TXHC, TNI, YKWK, NJS, TLV) for testing.
This forces the model to generalise to unseen voices — the actual task.
