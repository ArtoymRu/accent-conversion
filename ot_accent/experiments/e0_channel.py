"""E0. The channel gate: is WavLM -> HiFi-GAN resynthesis clean before anything is edited?

Compares the original test utterances with their resynthesis by both public
vocoders. Pass if WER and speaker cosine stay close to the original and the
files in results/e0_audio do not sound robotic. If this fails, no audio claim
downstream means anything.

    python -m experiments.e0_channel
"""
import numpy as np
import soundfile as sf

from experiments.common import load_frames, load_table, setup, show, write_csv
from otaccent import features, metrics


def main():
    cfg = setup(__doc__)
    assert not cfg.synthetic, "E0 is about audio; it has no synthetic version"
    tab = load_table(cfg, cfg.edit_layer)
    speakers = [tab.speakers[tab.idx(a)[0]] for a in np.unique(tab.accents)]     # one per accent
    vocoders = {"resynthesis (prematched vocoder)": features.Vocoder(cfg.device, prematched=True),
                "resynthesis (plain vocoder)": features.Vocoder(cfg.device, prematched=False)}
    asr, voice = metrics.Asr(cfg.device), metrics.SpeakerEncoder(cfg.device)
    audio = cfg.results / "e0_audio"
    audio.mkdir(exist_ok=True)

    refs, hyps, cos = [], {"original": []}, {}
    for spk in speakers:
        f = load_frames(cfg, spk)
        for u in np.flatnonzero(f.is_test):
            refs.append(str(f.texts[u]))
            hyps["original"].append(asr(f.wavs[u]))
            for name, vocoder in vocoders.items():
                out = vocoder(f.frames[f.utt == u])
                hyps.setdefault(name, []).append(asr(out))
                cos.setdefault(name, []).append(float(voice(out) @ voice(f.wavs[u])))
                if u == np.flatnonzero(f.is_test)[0]:
                    sf.write(audio / f"{spk}_{name.split('(')[1][:-1].replace(' ', '_')}.wav", out, features.SR)
            if u == np.flatnonzero(f.is_test)[0]:
                sf.write(audio / f"{spk}_original.wav", f.wavs[u], features.SR)
    rows = [dict(system=k, wer=asr.wer(refs, h), speaker_cosine=float(np.mean(cos.get(k, [1.0])))) for k, h in hyps.items()]
    show(rows, ["system", "wer", "speaker_cosine"])
    write_csv(cfg.results / "e0_channel.csv", rows)
    print(f"listen: {audio}")


if __name__ == "__main__":
    main()
