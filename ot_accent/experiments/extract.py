"""Audio -> cache. The only step that reads the corpora; run it once, on a GPU.

For every speaker it writes cache/<speaker>.npz with
  - sufficient statistics per (layer, half, phone) over the training utterances,
  - the frames, labels, text and waveform of the pool and test utterances.
Test utterances are the same sentences for all speakers and enter no statistic.
A speaker whose file exists is skipped, so an interrupted run can be resumed.

    python -m experiments.extract --config config.yaml
"""
from collections import defaultdict

import numpy as np

from experiments.common import save_speaker, setup
from otaccent import corpus, features
from otaccent.stats import TableBuilder


def main():
    cfg = setup(__doc__)
    cfg.cache.mkdir(parents=True, exist_ok=True)
    by_speaker = defaultdict(list)
    for u in corpus.utterances(cfg.l2_arctic, cfg.cmu_arctic):
        by_speaker[u.speaker].append(u)
    assert by_speaker, "no utterances found: check l2_arctic and cmu_arctic in the config"
    shared = sorted(set.intersection(*(set(u.utt for u in us) for us in by_speaker.values())))
    test, pool = set(shared[-cfg.test_utts:]), set(shared[-cfg.test_utts - cfg.pool_utts:-cfg.test_utts])
    print(f"{len(by_speaker)} speakers, {len(shared)} shared sentences, test {min(test)}..{max(test)}")

    encoder = features.Encoder(cfg.device)
    for speaker, utts in sorted(by_speaker.items()):
        out = cfg.cache / f"{speaker}.npz"
        if out.exists():
            continue
        train = [u for u in utts if u.utt not in test][:cfg.max_utts]
        keep = [u for u in utts if u.utt in pool or u.utt in test]
        builders = {l: TableBuilder(len(corpus.PHONES), 1024) for l in cfg.layers}
        frames, labels, texts, wavs, is_test = [], [], [], [], []
        for i, u in enumerate(train + [u for u in keep if u not in train]):
            wav = features.load_audio(u.wav)
            feats = encoder(wav, cfg.layers)
            lab = features.frame_labels(u.phones(), len(feats[cfg.edit_layer]))
            if u.utt not in test:
                for l in cfg.layers:
                    builders[l].add(speaker, u.accent, i % 2, feats[l], lab)
            if u in keep:
                frames.append(feats[cfg.edit_layer]), labels.append(lab), texts.append(u.text())
                is_test.append(u.utt in test), wavs.append(wav if u.utt in test else np.zeros(0))
        stats = {l: builders[l].rows[speaker][1:] for l in cfg.layers}
        save_speaker(out, utts[0].accent, cfg.layers, stats, frames, labels, is_test, texts, wavs)
        print(f"{speaker:6s} {utts[0].accent:10s} {len(train)} utterances")


if __name__ == "__main__":
    main()
