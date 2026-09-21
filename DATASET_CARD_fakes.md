# codeswitch-deepfake-fakes

Synthetic counterpart to `codeswitch-deepfake-real`. One fake per real utterance,
15,961 files, 16 kHz mono FLAC.

## Attack systems

| Attack | Model | Architecture | Coverage |
|---|---|---|---|
| A | XTTS-v2 | autoregressive tokens + HiFi-GAN | Hindi, Hinglish |
| B | FreeVC (freevc24) | VITS flow + HiFi-GAN over WavLM content | all three conditions |

A third system is held out entirely and never appears here, so attack generalisation can be
measured against data no training run has seen.

Bengali is attack B only, because XTTS does not support `bn`. Comparisons involving Bengali
are restricted to the FreeVC subset.

Transcripts containing digits are always attack B, regardless of the even
(set, split, speaker) split applied elsewhere. XTTS expands numbers through num2words, which
has no Hindi converter and raises rather than falling back. All 86 such utterances fall in the
code-switched test set, which therefore carries 1,609 B against 1,454 A rather than an even
split.

## Controls

Speakers are matched within condition and split. An XTTS fake is voiced by the same speaker
and speaks the same transcript as the real file it pairs with, so synthesis is the only
difference. A FreeVC fake converts a real utterance towards another speaker of the same
condition and split, cycled so the fake-side identity distribution matches the real side.

Every file passed through native -> 8 kHz -> 16 kHz, identical to the real audio, which was
frozen through that path and holds no energy above 4 kHz. Without it, the raw 24 kHz output
of both generators carried roughly 2.3e6 times the real files' energy above 4 kHz.

## Known characteristics

XTTS output runs about 9% shorter than its paired real file on code-switched transcripts and
about 10% longer on monolingual ones. This is not content loss: binning the code-switched set
by Latin-script share leaves the ratio flat at ~0.91 whatever proportion of the transcript is
English, and listening confirms the English words are present. FreeVC sits at 0.996
throughout, since conversion cannot alter timing.

Above 4 kHz both attacks sit below the reals after the bottleneck, XTTS by a factor of 4,
FreeVC by 8, against a real median of 8.3e-09. Both are within the one-decade tolerance this
dataset was built to, but a detector reaching implausible accuracy should be tested against
that single feature before the result is believed.

## Provenance

Source speech: MUCS / OpenSLR 104. Seed 1337. Splits inherited from the real dataset and
not recomputed. Generation notebook: `07_generate_fakes.py`.

## Licences

XTTS-v2 is CPML, non-commercial; academic research is within scope. FreeVC is MIT, and its
checkpoint is trained on VCTK (CC BY 4.0), which carries an attribution requirement. Neither
generator is shipped in the deployed demo, which runs the detector only.
