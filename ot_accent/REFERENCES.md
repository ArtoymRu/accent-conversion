# References

Every work the project leans on, with its paper, its code, and how well we
actually know it. Keep this honest: update the last column when you read
something properly.

**Read:** `full` read in full · `summary` full text seen through an automatic
summary · `abstract` abstract or snippets only · `code` repository read ·
`unread` cited on someone's word.
**ID:** identifiers marked *(memory)* were written from memory and must be
checked before they go into a bibliography.
Repository links were checked on 12 Nov 2026 unless noted.

## Closest work

| Work | Paper | Code | Why it matters | Read |
|---|---|---|---|---|
| **PACE.** Qu, Panariello, Todisco, Evans. *Partial Accent-Control Editing in Frozen Speech Representations for Accent Conversion.* 2026 | [arXiv:2609.22031](https://arxiv.org/abs/2609.22031) | [eurecom-asp/PACE-AC](https://github.com/eurecom-asp/PACE-AC): a README only, no code | Same channel (WavLM-Large L6, kNN-VC vocoder) and data (L2-ARCTIC). Global PLS subspace, entropic transport, retrieval fusion. Main baseline; states our direction as future work | summary |
| **StateVC.** Qu, Panariello, Todisco, Evans. *Shared-State Local Translations for Training-Free Voice Conversion.* 2026 | [arXiv:2610.01952](https://arxiv.org/abs/2610.01952) | [eurecom-asp/StateVC](https://github.com/eurecom-asp/StateVC) (complete), [samples](https://github.com/eurecom-asp/StateVC_listening_samples) | Per-state mean shifts in WavLM space with soft posteriors and a strength: the same class of map as `conditional shift`, used for voice conversion | abstract, code README |
| Qu, Todisco, Evans. *Phoneme-Aware Pronunciation Representations for L2-English L1-Background Accent Identification.* 2026 | [arXiv:2609.10466](https://arxiv.org/abs/2609.10466) | none found | Speaker-disjoint accent identification from phoneme-aligned units. Prior art for "accent is phoneme-dependent"; a candidate independent accent metric | abstract |
| Qu, Panariello, Todisco, Evans. *Multi-sample Synthetic Supervision for Accent Conversion.* 2026 | [arXiv:2610.01961](https://arxiv.org/abs/2610.01961) | not looked for | Trained accent conversion from the same group, six target accents, with listening ratings | abstract |

## The channel

| Work | Paper | Code | Why it matters | Read |
|---|---|---|---|---|
| **kNN-VC.** Baas, van Niekerk, Kamper. *Voice Conversion With Just Nearest Neighbors.* Interspeech 2023 | arXiv:2305.18975 *(memory)* | [bshall/knn-vc](https://github.com/bshall/knn-vc) | Source of our encoder checkpoint and of the HiFi-GAN that inverts WavLM layer 6; the barycentric map | code |
| **WavLM.** Chen et al. IEEE JSTSP 2022 | arXiv:2110.13900 *(memory)* | [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) | The frozen encoder | unread |

## Transport and conversion in SSL feature space

| Work | Paper | Code | Why it matters | Read |
|---|---|---|---|---|
| **MKL-VC.** Lobashev, Yermekova, Larchenko. *Training-Free Voice Conversion with Factorized Optimal Transport.* Interspeech 2025 | [arXiv:2506.09709](https://arxiv.org/abs/2506.09709) | [alobashev/mkl-vc](https://github.com/alobashev/mkl-vc) | Gaussian (Monge–Kantorovich linear) transport map in WavLM space, for voice | abstract |
| Cai et al. *Universal Speech Content Factorization.* 2026 | [arXiv:2603.08977](https://arxiv.org/abs/2603.08977) | linked in the paper (anonymised) | Linear relations between speakers' WavLM features; training-free conversion | abstract |
| Selitskiy, Kocharekar. *Discrete Optimal Transport and Voice Conversion.* 2025 | arXiv:2505.04382 | not looked for | Transport in SSL space for voice | unread |
| Peyré, Cuturi. *Computational Optimal Transport.* 2019 | arXiv:1803.00567 *(memory)* | [PythonOT/POT](https://github.com/PythonOT/POT) | Background; Gaussian closed form; Sinkhorn | unread for this project |
| Flamary, Lounici, Ferrari. *Concentration bounds for linear Monge mapping estimation.* 2019 | arXiv:1905.10155 *(memory)* | in POT | How many samples a linear map needs | unread |
| Hosseini, Hsu, Taghvaei. *Conditional Optimal Transport on Function Spaces.* 2023 | arXiv:2311.05672 *(memory)* | — | Conditional transport, the formal frame of section 3.5 | unread |
| Bunne, Krause, Cuturi. *Supervised Training of Conditional Monge Maps.* NeurIPS 2022 | arXiv:2206.14262 *(memory)* | — | Same | unread |

## Structure of speech representations

| Work | Paper | Code | Why it matters | Read |
|---|---|---|---|---|
| Liu, Tang, Goldwater. *Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces.* Interspeech 2023 | [arXiv:2305.12464](https://arxiv.org/abs/2305.12464) | not looked for | Speaker and phone subspaces nearly orthogonal, for APC and CPC only. Motivation for the additive speaker term, not evidence for WavLM | abstract, introduction |
| *Orthogonality and isotropy of speaker and phonetic information in self-supervised speech representations.* 2024 | to locate | — | May extend the above to more models. Title seen in a listing only | unread |
| Yang et al. *What Can an Accent Identifier Learn? Probing Phonetic and Prosodic Information in a Wav2vec2-based Accent Identification Model.* 2023 | [arXiv:2306.06524](https://arxiv.org/abs/2306.06524) | — | Accent-specific phoneme representations in a middle layer, on L2-ARCTIC | abstract |
| Segment-level SSL distances and perceived accentedness, Hindi English. 2025 | [arXiv:2506.17542](https://arxiv.org/abs/2506.17542) (title to fill) | — | Segmental accent in SSL representations related to human ratings | snippet |
| Zhong, Wang, Richmond, Bell. *Rethinking Discrete Speech Representation Tokens for Accent Generation.* 2026 | [arXiv:2601.19786](https://arxiv.org/abs/2601.19786) | — | How much accent discrete tokens keep; an accent ABX task | abstract |

## Accent conversion and control

| Work | Paper | Code | Why it matters | Read |
|---|---|---|---|---|
| **AccentFlow.** Ostrovskii, Sheremet, Karpov. SPECOM 2026 | LNCS 16935 | [ArtoymRu/accent_flow](https://github.com/ArtoymRu/accent_flow) | Origin of the question: one delta vector in speaker-embedding space. Ideas only; its numbers are not used here | own work |
| **TokAN.** Bai et al. *Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data.* 2025 | arXiv:2507.17735 | [P1ping/TokAN](https://github.com/P1ping/TokAN) | Trained accent normalisation; a possible generative reference | unread |
| **DLM-AN.** Bai et al. *Controllable Accent Normalization via Discrete Diffusion.* 2026 | [arXiv:2603.14275](https://arxiv.org/abs/2603.14275) | [P1ping/DLM-AN](https://github.com/P1ping/DLM-AN) | Controllable strength in a trained system | abstract |
| Sun et al. *Activation Steering for Accent Adaptation in Speech Foundation Models.* 2026 | [arXiv:2603.05813](https://arxiv.org/abs/2603.05813) | — | Global mean-shift accent directions per layer, for ASR | snippet |
| Yang, Hansen. *Activation Steering for Accent-Neutralized Zero-Shot Text-To-Speech.* 2026 | [arXiv:2603.05977](https://arxiv.org/abs/2603.05977) | — | Steering inside a synthesiser | snippet |
| *FAC-FACodec: Controllable Zero-Shot Foreign Accent Conversion with Factorized Speech Codec.* 2025 | [arXiv:2510.10785](https://arxiv.org/abs/2510.10785) | — | Strength control with an explicit identity trade-off | snippet |
| DART and MLV | take from PACE's bibliography | — | The two trained baselines PACE compares with | unread |
| Bourrel, Issam, Spanakis. *A Probe Shift Is Not a Fairness Fix.* 2026 | arXiv:2609.18533 | — | A warning that moving a probe is not moving perception | unread |

## Metrics and tools

| Tool | Paper | Code | Used for |
|---|---|---|---|
| Whisper | arXiv:2212.04356 *(memory)* | [openai/whisper](https://github.com/openai/whisper) | word error rate (`metrics.Asr`) |
| Resemblyzer (GE2E) | arXiv:1710.10467 *(memory)* | [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | speaker cosine, comparable to PACE's SECS (`metrics.Resemblyzer`) |
| ECAPA-TDNN | arXiv:2005.07143 *(memory)* | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | speaker cosine (`metrics.Ecapa`) |
| PACE's accent classifier | linked in a footnote of PACE | to locate | reporting accent on their scale |

## Data

| Corpus | Paper | Where | Notes |
|---|---|---|---|
| L2-ARCTIC | Zhao et al., Interspeech 2018 | [psi.engr.tamu.edu/l2-arctic-corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/) | 24 speakers, 6 first languages; forced alignments in `textgrid/` |
| CMU ARCTIC | Kominek, Black, 2004 | [festvox.org/cmu_arctic](http://festvox.org/cmu_arctic/) | optional `native` group; other studio, other aligner |

## Venue

Interspeech 2027, São Paulo, 29 Aug – 2 Sep 2027. Paper deadline 9 Feb 2027
according to a tracker citing
[interspeech2027.org/important-dates](https://interspeech2027.org/important-dates/);
that page has not been opened by us.

## To do in this file

- [ ] Read PACE, StateVC, the phoneme-aware identification paper and MKL-VC in full; change their status.
- [ ] Find PACE's classifier checkpoint and the DART and MLV references.
- [ ] Check every identifier marked *(memory)*.
- [ ] Locate the 2024 orthogonality paper and arXiv:2506.17542's title.
