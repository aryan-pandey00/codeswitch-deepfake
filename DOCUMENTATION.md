# CodeSwitch-Deepfake: full documentation

**Audio deepfake detection for code-switched Indian speech (Hindi-English and Bengali-English),
and a measurement of why these detectors fail here.**

This is the complete technical account: every dataset decision, every experiment with its
numbers, the things that nearly went wrong, and how the demo is deployed.
[`README.md`](README.md) is the short version: start there for the finding without the working.

The short version: deepfake voice detectors do break on Indian speech, but not for the reason
the field assumed. They do not break because of the language, and they do not break by letting
fakes through. They break by rejecting real human beings, roughly half of them, because they
learned what the training microphone sounded like instead of what synthetic speech sounds like.
Adding eight speakers from a new recording setup to the training data takes that error from
29-85% down to 0.1-0.3%.

This repository holds the dataset builders, the four detectors, every evaluation, and a live
demo that scores one clip twice so the difference is visible on a single upload.

## Contents

1. [What this project is, in plain words](#1-what-this-project-is-in-plain-words)
2. [The vocabulary, explained](#2-the-vocabulary-explained)
3. [The finding, in numbers](#3-the-finding-in-numbers)
4. [How the dataset was built](#4-how-the-dataset-was-built)
5. [The four detectors](#5-the-four-detectors)
6. [What was measured, experiment by experiment](#6-what-was-measured-experiment-by-experiment)
7. [Things that nearly went wrong](#7-things-that-nearly-went-wrong)
8. [Limitations, stated plainly](#8-limitations-stated-plainly)
9. [The demo, from upload to verdict](#9-the-demo-from-upload-to-verdict)
10. [Repository layout](#10-repository-layout)
11. [The notebooks, in order](#11-the-notebooks-in-order)
12. [Reproducing this](#12-reproducing-this)
13. [Data, licences and ethics](#13-data-licences-and-ethics)

## 1. What this project is, in plain words

### The thing being built

A **deepfake voice detector**. Software that listens to a voice clip and answers one question:
did a human being say this, or did a machine generate it?

This matters because voice cloning is now cheap and good. A few seconds of someone's recorded
speech is enough to make them say anything, and the obvious uses are fraud, the "your son is in
trouble, send money" phone call, and fabricated evidence. The defence is a detector, and
detectors are only useful if they work on the audio people actually have: a phone, in a room, in
the language people actually speak.

### What people speak in India, and why that is a problem

Most Indians do not speak one language at a time. A single sentence casually mixes Hindi and
English, and the same happens with Bengali, Tamil, and everything else. Linguists call this
**code-switching**. It is not broken speech or slang, it is the normal register for hundreds of
millions of people.

Nearly every published deepfake detector is built and tested on clean English studio recordings.
Take one to Indian speech and it falls apart. The assumed reason, including in the one prior
project found on this exact topic, was the language itself: either Indian languages in general,
or the mixing.

**That was the starting hypothesis of this project too, and this project disproved it.**

### What was built to test it

A dataset where everything is held fixed except the one variable under study.

One corpus, the same speakers, the same rooms, the same microphones, the same recording sessions.
Inside it sit code-switched Hindi-English utterances, code-switched Bengali-English utterances,
and plain monolingual utterances from the same people in the same sessions. The fake half of the
dataset was generated here by cloning those exact speakers with two different synthesis systems.

That last control is the part nobody else had. If a detector is tested on switched speech from
one collection and monolingual speech from another collection, then language and recording setup
change together, and no conclusion about either is possible. Holding the corpus fixed separates
them.

### What came out

**Language is not the cause.** Feed the detector plain monolingual Hindi with zero mixing,
recorded in a different session: it rejects about half of it. Feed it plain monolingual Hindi
from the training session: it rejects none of it. Same language, same content type, opposite
outcome. What changed was the room and the microphone.

**The failure has a strange shape, and this is the part that matters.** The detector never
misses a fake. Not one, across every out-of-corpus condition. What it does instead is reject
real people. Deployed in India, this does not leak deepfakes, it accuses innocent speakers of
being synthetic, about half the time.

**The result that was not expected.** Three model designs were trained. The more a model was
allowed to tune itself to the training recordings, the worse this got: a frozen model with a
tiny classifier on top rejects 29.2% of genuine speech, a classic CNN-BiLSTM rejects 65.7%, and
a fine-tuned version of the frozen model rejects 84.7%. Fine-tuning is the field's default
recipe. **The default recipe nearly triples the damage.**

**The obvious escape is closed.** The natural reply is "just move the decision threshold". That
was tested in the most generous form possible, a threshold tuned using the answer key for the
entire target corpus, which no real deployment could ever have. It still leaves 44.6% and 18.4%
of genuine speech rejected for the two fitted models. The problem lives inside the model's
internal representation, not in where the cutoff sits.

**The fix is cheap and it generalises.** Add eight speakers from the new recording setup to
training, about 7% more data. Rejection falls to 0.1-0.3%, and costs nothing on the original
data. Better: adding *Hindi-English* speakers also repairs *Bengali-English*. The model is not
memorising one recording chain, it is learning to tolerate recording chains in general. That is
what makes it a finding rather than a patch.

### The one-sentence version

> Deepfake detectors fail on Indian speech, but not because of the language, which is what
> everyone assumed. They fail because of the recording conditions, they fail by rejecting real
> people rather than by missing fakes, the standard way of building them makes it three times
> worse, and eight extra speakers fixes it.

## 2. The vocabulary, explained

Everything in the sections below uses these terms. Nothing here assumes prior knowledge.

### About the audio

| Term | What it means |
|---|---|
| **TTS (text to speech)** | Software that reads written text out loud in a chosen voice. XTTS-v2 here. |
| **Voice conversion (VC)** | Software that takes a real recording and repaints it in someone else's voice, keeping the original timing and delivery. FreeVC here. |
| **Vocoder** | The final stage of almost every speech generator: it turns a compact description of sound into an actual waveform. HiFi-GAN is the common one, and both generators used here end in it. |
| **Sample rate** | How many times per second the audio was measured. 16 kHz means 16,000 measurements per second. |
| **Bandwidth / the 4 kHz ceiling** | Audio sampled at 8 kHz can physically carry no sound above 4 kHz, which is why old phone calls sound muffled. Anything above that ceiling is gone forever, even after upsampling back to 16 kHz. |
| **Code-switching** | Mixing two languages inside one sentence. "Mujhe ek meeting attend karni hai." |
| **Corpus** | A collection of recordings gathered as one project, with its own microphones, rooms, protocol and speakers. |
| **Channel / recording condition** | Everything between the mouth and the file: the room, the microphone, the phone, the codec. Two people saying the same words in two rooms produce very different signals. |
| **Codec** | Compression applied to speech to make it small enough to transmit. WhatsApp, GSM calls and VoIP each use different ones and each damages the audio differently. |

### About the models

| Term | What it means |
|---|---|
| **Self-supervised model** | A large model pretrained on huge amounts of raw speech with no labels, learning the structure of speech in general. XLS-R here, trained on 436,000 hours across 128 languages. |
| **XLS-R** | The specific self-supervised speech model used as the front end, `facebook/wav2vec2-xls-r-300m`, about 300 million parameters. |
| **Frontend / representation** | The part that converts raw audio into numbers that describe it. Everything downstream sees only those numbers. |
| **Frozen** | The frontend's internal weights are never changed by this project's training. It describes the audio exactly as it did before, and only a small classifier on top learns anything. |
| **Linear probe** | The smallest possible classifier, a weighted sum of the frontend's numbers. 2,048 weights here. If it separates the classes, the information was already in the representation. |
| **Fine-tuning** | The opposite of frozen: letting the big model itself change to fit the training data. Usually improves in-domain scores. Here it makes transfer far worse. |
| **LoRA** | A cheap way to fine-tune. Instead of moving all 300 million weights, small adapter matrices are inserted and only those are trained, 657,409 parameters here. |
| **MFCC** | A compact, decades-old summary of a sound's spectrum, 40 numbers per frame here. The standard input before self-supervised models arrived. |
| **CNN-BiLSTM** | The pre-2020 standard detector architecture. A convolutional network finds local patterns in the spectrogram, a bidirectional LSTM reads the sequence forwards and backwards. 504,321 parameters, no pretraining. |
| **AASIST** | A published state-of-the-art detector, released with weights its own authors trained on the ASVspoof2019 benchmark. Used here exactly as released, never trained. |
| **Zero-shot** | Running a model on new data with no training on that data at all. |
| **Quantisation / int8** | Storing the model's numbers in 8 bits instead of 32 to make it four times smaller and cheaper to host. It introduces a small error, which is measured here rather than assumed harmless. |
| **ONNX** | A portable model file format that runs without the original training framework, which is what lets the demo stay small. |

### About the measurements

Throughout, **the positive class is "fake"**. So the two errors have names worth keeping straight,
because this entire project turns on which one moved.

| Term | What it means |
|---|---|
| **FAR, false acceptance rate** | A fake got through and was called genuine. The error everyone worries about. |
| **FRR, false rejection rate** | A real human was called fake. The error that actually happened here. |
| **Threshold** | The cutoff on the model's score that turns a number into a verdict. Above it, fake. |
| **Calibration** | Whether the threshold sits in the right place for the data being seen. |
| **Discrimination** | Whether the model can tell the two classes apart at all, regardless of where the cutoff sits. A model can have perfect discrimination and useless calibration at the same time, and that is exactly what happens below. |
| **AUC** | A threshold-free score for discrimination. 100 is perfect, 50 is a coin flip. |
| **EER, equal error rate** | The error rate at the cutoff where the two errors are equal. Threshold-free, so it measures discrimination, not deployment behaviour. Lower is better. |
| **Accuracy, precision, recall, F1** | The familiar summary metrics. Reported here because readers look for them, but note that accuracy hides everything interesting when the two errors are wildly asymmetric. |
| **In domain** | Tested on data from the same corpus the model trained on. |
| **Out of corpus** | Tested on a different corpus. This is where everything falls apart. |
| **Speaker-disjoint** | No person appears in both the training set and the test set, so nothing can be won by memorising voices. |
| **Confound** | A second difference between the two classes that has nothing to do with the thing being studied, so the model can win by reading that instead. Most of the engineering effort in this project went into finding and removing these. |
| **Oracle** | A deliberately unfair best case, given information a real system could never have, used to prove that even that is not enough. |

## 3. The finding, in numbers

All rows below are the frozen XLS-R probe, trained only on monolingual Hindi. Genuine speech
rejected, at the deployed threshold:

| Test condition | Genuine speech rejected |
|---|---|
| Monolingual Hindi, training corpus, speakers never seen before | **0.0%** |
| Monolingual Hindi, the corpus's official test partition | **0.0%** |
| Monolingual Hindi, recorded inside the Hindi-English corpus | **51.5%** |
| Code-switched Hindi-English | **49.2%** |
| Monolingual Bengali, recorded inside the Bengali-English corpus | **23.0%** |
| Code-switched Bengali-English | **14.4%** |

Read rows 1 and 3 together. Same language, same monolingual content, zero code-switching, unseen
speakers in both cases. The only thing that changed is which recording session the audio came
from, and rejection goes from 0.0% to 51.5%.

Now read rows 3 and 4. Monolingual speech is rejected slightly *more* than the code-switched
speech sitting beside it in the same corpus. The same holds in Bengali, 23.0% against 14.4%.
Switching is not the axis.

And the other half of the story, across every out-of-corpus condition tested:

| | |
|---|---|
| Fakes missed (false acceptance) | **0.00%** |
| Genuine speech rejected (false rejection), pooled | **29.2%** |

**Every single fake was caught. Nearly a third of real people were not.** The decision boundary
this detector learned is not "synthetic versus genuine", it is "matches my training channel
versus does not". Fakes differ from that channel, so they are always caught. Genuine speech from
another session differs too, so a large share of it is caught as well.

## 4. How the dataset was built

**15,961 real utterances and 15,961 fake ones**, one fake per real file, 16 kHz mono FLAC.

### The real speech

Cut from MUCS / OpenSLR [103](https://openslr.org/103/) (monolingual Indian languages) and
[104](https://openslr.org/104/) (code-switched Hindi-English and Bengali-English), both CC BY-SA
4.0. Using one corpus family for everything is the whole point: the code-switched sets contain
monolingual utterances from the same speakers in the same sessions, which is the control that
separates language from recording condition.

Training uses monolingual Hindi only, 11,410 utterances. The code-switched corpora are the target,
never seen during baseline training.

### The fakes

Generated here, because the point is to hold everything except synthesis constant.

| Attack | System | How it works | Covers |
|---|---|---|---|
| A | XTTS-v2 | Text to speech: autoregressive tokens, then a HiFi-GAN vocoder | Hindi, Hindi-English |
| B | FreeVC (freevc24) | Voice conversion: VITS flow over WavLM content features, then HiFi-GAN | all three conditions |

A third system is held out entirely and never appears in the data, so attack generalisation can
be measured against something no training run has seen. Bengali is attack B only, because XTTS
does not support it.

Two matching rules make the two classes separable only by synthesis:

- **Speakers are matched within condition and split.** An XTTS fake is voiced by the same speaker,
  saying the same transcript, as the real file it pairs with. A FreeVC fake converts a real
  utterance towards another speaker from the same condition and split. Cloning the code-switched
  fakes from the monolingual speaker pool would have put voices in one class that appear in
  neither the other class nor the same condition, and the detector would have learned identity.
- **Every file, real and generated, passes through the same bottleneck.** Explained next, and it
  is the single most important design decision in the project.

### The bandwidth trap, and why every file was deliberately degraded

The monolingual recordings are natively 8 kHz. The code-switched recordings are natively 16 kHz.
That split falls **exactly** on the train/test boundary.

Left alone, this ruins the entire experiment. Every result would have been explainable as "the
test audio simply contains higher frequencies than the training audio", which is a statement
about sample rates and not about language, channel, or deepfakes.

So every file, real and fake, is resampled native to 8 kHz to 16 kHz. The trip down to 8 kHz
throws away everything above 4 kHz permanently, and the trip back up restores the file format
without restoring the content. After it, no file in the dataset has meaningful energy above
4 kHz, and bandwidth is no longer a variable.

This also closed a second problem. Raw 24 kHz generator output carried roughly **2.3 million
times** the real clips' energy above 4 kHz, which is a giveaway a detector would happily learn
instead of learning synthesis. After the bottleneck both attacks sit within one decade of the
real audio.

### Counts

| | |
|---|---|
| Real utterances | 15,961, 0.82 GB |
| Fake utterances | 15,961, one per real file |
| Training set (monolingual Hindi) | 11,410 utterances |
| Hindi-English speakers evaluated | 22 (of 30, the other 8 are reserved for the fix) |
| Bengali-English speakers evaluated | 32 (of 40, same reason) |
| Everything | 16 kHz mono FLAC, no energy above 4 kHz |

## 5. The four detectors

Four detectors were measured, which answers the obvious objection to a one-model result: *maybe a
better model just would not have this problem.*

| Detector | What it is | Trained here? | Capacity fitted to the training corpus |
|---|---|---|---|
| **XLS-R + linear probe** (baseline) | Frozen self-supervised frontend, 2,048-weight logistic head on pooled features | head only | none |
| **CNN-BiLSTM** | 40 MFCCs, convolutions plus a bidirectional LSTM, 504,321 parameters, no pretraining | fully | 504k parameters from scratch |
| **XLS-R + LoRA** | Same frontend, but adapters inside the query and value projections of all 20 retained blocks, 657,409 trainable parameters | frontend and head | the encoder itself moves |
| **AASIST** | Published state of the art, graph attention over raw waveform, 297k parameters, the authors' own ASVspoof2019 weights | no, zero-shot | not applicable |

The first three share no training recipe and, in the CNN-BiLSTM's case, no component at all with
the others. They were given identical rows, identical splits, and the same speakers held out.

## 6. What was measured, experiment by experiment

### 6.1 The full metric set

Positive class is fake, so FAR is a fake accepted and FRR is a real person rejected. `baseline` is
trained on monolingual Hindi only. `fixed` adds eight speakers per target corpus. **Every number
excludes those eight speakers**, which is what makes each pair of rows comparable.

| Model | Condition | n | Acc | Prec | Rec | F1 | AUC | EER | FAR | FRR (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | in-domain dev, unseen speakers | 2,640 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.0 (0.0-0.3) |
| baseline | in-domain control, official test partition | 3,276 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.0 (0.0-0.2) |
| baseline | Hindi-English, all | 4,548 | 75.2 | 66.8 | 100.0 | 80.1 | 99.73 | 2.42 | 0.00 | **49.6 (47.6-51.7)** |
| baseline | Hindi-English, monolingual only | 846 | 74.2 | 66.0 | 100.0 | 79.5 | 99.66 | 3.19 | 0.00 | **51.5 (46.8-56.3)** |
| baseline | Hindi-English, code-switched only | 3,702 | 75.4 | 67.0 | 100.0 | 80.3 | 99.75 | 2.32 | 0.00 | 49.2 (46.9-51.5) |
| baseline | Bengali-English, all | 6,746 | 92.3 | 86.6 | 100.0 | 92.8 | 99.91 | 0.79 | 0.00 | 15.5 (14.3-16.7) |
| baseline | Bengali-English, monolingual only | 844 | 88.5 | 81.3 | 100.0 | 89.7 | 99.51 | 2.13 | 0.00 | **23.0 (19.2-27.2)** |
| baseline | Bengali-English, code-switched only | 5,902 | 92.8 | 87.4 | 100.0 | 93.3 | 99.95 | 0.66 | 0.00 | 14.4 (13.2-15.7) |
| baseline | out of corpus, pooled | 11,294 | 85.4 | 77.4 | 100.0 | 87.2 | 99.82 | 1.82 | 0.00 | **29.2 (28.1-30.4)** |
| fixed | in-domain dev, unseen speakers | 2,640 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.0 (0.0-0.3) |
| fixed | in-domain control, official test partition | 3,276 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.0 (0.0-0.2) |
| fixed | Hindi-English, all | 4,548 | 99.9 | 99.8 | 100.0 | 99.9 | 100.00 | 0.00 | 0.00 | 0.22 (0.09-0.51) |
| fixed | Hindi-English, monolingual only | 846 | 99.5 | 99.1 | 100.0 | 99.5 | 100.00 | 0.00 | 0.00 | 0.95 (0.37-2.41) |
| fixed | Hindi-English, code-switched only | 3,702 | 100.0 | 99.9 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.05 (0.01-0.31) |
| fixed | Bengali-English, all | 6,746 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.03 (0.01-0.17) |
| fixed | Bengali-English, monolingual only | 844 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.0 (0.0-0.90) |
| fixed | Bengali-English, code-switched only | 5,902 | 100.0 | 100.0 | 100.0 | 100.0 | 100.00 | 0.00 | 0.00 | 0.03 (0.01-0.19) |
| fixed | out of corpus, pooled | 11,294 | 99.9 | 99.9 | 100.0 | 99.9 | 100.00 | 0.00 | 0.00 | **0.11 (0.05-0.23)** |

One line: **out of corpus, accuracy 85.4% to 99.9% and false rejection 29.2% to 0.11%, with no
fake missed by either head and no in-domain cost.**

### 6.2 Code-switching contributes nothing, tested four ways

The monolingual rows above are the cleanest evidence, but the null holds under every other test
applied to it.

**Dose response.** If switching caused the damage, more switching should cause more damage.
Rejection is flat across switching intensity, and the *worst* bin is the one with no switching
at all:

| bin | median English word ratio | median switching index | rejected |
|---|---|---|---|
| monolingual | 0.00 | 0.0 | 49.7% |
| low | 0.08 | 7.7 | 54.2% |
| mid | 0.18 | 18.2 | 50.7% |
| high | 0.38 | 37.5 | 52.2% |
| very high | 0.62 | 38.5 | 53.8% |

**Regression.** Score against switch rate, controlling for English word ratio and utterance
length, with speaker-clustered standard errors, genuine utterances only: null at every layer in
both corpora (Hindi-English layer 20: coefficient -0.227, p = 0.753). The single significant
coefficient anywhere is *negative*, meaning more switching looked slightly more genuine.

**The cross-corpus check.** Bengali-English is *more* code-switched than Hindi-English (87.8% of
utterances against 80.0%) and further from the training language, yet it degrades roughly three
times less. Inconsistent with either switching or language distance as the cause.

**Direction instability.** Across the two ways of slicing the same data, monolingual is rejected
more than switched in three of four comparisons and less in one. A real effect does not change
sign between corpora or between speaker subsets. This is what a null looks like, and it is
reported that way rather than as a reverse effect.

### 6.3 The failure is calibration, not discrimination

There is an apparent contradiction in the table above, and resolving it sharpens the whole claim.
On Hindi-English the baseline holds **99.73 AUC and 2.42% EER**, which is near-perfect ranking,
while rejecting **49.6%** of genuine speech. Both are true at once.

The reason is that the genuine score distribution slides upward until its median lands on the
decision boundary:

| Corpus | genuine median score | fake median score | deployed threshold |
|---|---|---|---|
| Hindi-English | **-0.03** | +9.81 | 0.00 |
| Bengali-English | -2.77 | +9.30 | 0.00 |

A genuine median of -0.03 against a boundary of 0.00 is what a 50% rejection rate looks like from
the inside. Nudge the population by a hair and half of it crosses. The *ordering* survives the
domain change. The *calibration* does not.

### 6.4 So why not just move the threshold?

Because it was tried, in the most generous form that exists, and it is not enough.

The oracle threshold below is fitted on the **entire target corpus with its true labels**. No
deployment has that. It is far more target information than the eight speakers the fixed head was
given. It is a deliberately unfair comparison, designed to lose.

| Model | Corpus | Oracle threshold | Retrained on 8 speakers, deployed threshold |
|---|---|---|---|
| XLS-R probe | Hindi-English | FRR 2.42% @ FAR 2.42% | **FRR 0.22% @ FAR 0.00%** |
| XLS-R probe | Bengali-English | FRR 0.80% @ FAR 0.80% | **FRR 0.03% @ FAR 0.00%** |
| CNN-BiLSTM | Hindi-English | FRR **44.55%** @ FAR 7.56% | **FRR 0.31% @ FAR 2.64%** |
| CNN-BiLSTM | Bengali-English | FRR **29.97%** @ FAR 0.27% | **FRR 0.24% @ FAR 0.06%** |
| XLS-R + LoRA | Hindi-English | FRR **18.43%** @ FAR 18.29% | **FRR 0.44% @ FAR 0.22%** |
| XLS-R + LoRA | Bengali-English | FRR 7.92% @ FAR 7.80% | **FRR 0.15% @ FAR 0.00%** |

**Retraining on eight speakers beats an oracle threshold fitted on the whole target corpus, for
every architecture.** On the frozen probe it wins by about eleven times. On the two fitted models
the oracle barely helps at all, because their equal error rate has itself collapsed, and EER is
the floor any threshold can reach.

The intuition: sliding a threshold trades one error for the other along a fixed score axis. Only
the representation can move that axis. For the CNN-BiLSTM the two score distributions have
literally merged out of corpus, genuine and fake both sitting at 0.58 to two decimals, so there
is nothing left for any threshold to separate.

### 6.5 The failure is architecture-independent, and adaptation makes it worse

Out of corpus, pooled over both target corpora:

| Model | Capacity fitted to the training corpus | Acc | AUC | EER | FAR | FRR |
|---|---|---|---|---|---|---|
| XLS-R + linear probe, baseline | none | 85.4 | 99.82 | 1.82 | 0.00 | **29.2** |
| CNN-BiLSTM, baseline | 504k params, from scratch | 67.1 | 94.31 | 13.39 | 0.07 | **65.7** |
| XLS-R + LoRA, baseline | 657k adapters steering the encoder | 57.6 | 93.54 | 13.79 | 0.02 | **84.7** |
| XLS-R + linear probe, fixed | none | 99.9 | 100.00 | 0.00 | 0.00 | 0.11 |
| CNN-BiLSTM, fixed | 504k params, from scratch | 99.3 | 99.95 | 0.60 | 1.10 | 0.27 |
| XLS-R + LoRA, fixed | 657k adapters steering the encoder | 99.8 | 100.00 | 0.18 | 0.09 | 0.27 |

**The signature is identical in all three.** Every fake caught, a large share of genuine speech
rejected. False acceptance never exceeds 0.07% while false rejection runs from 29% to 85%. So the
effect is not a quirk of self-supervised pretraining, of freezing, or of a linear head. No
architecture here learned a model of synthesis.

**The ordering is the finding.** Rejection tracks how much capacity was allowed to fit the
training corpus: none 29.2%, a small network trained on it 65.7%, a large pretrained encoder
steered towards it 84.7%. Fine-tuning the frontend on a single corpus nearly triples the false
rejection rate against leaving it frozen, which runs directly against standard practice in this
field.

In domain the ordering reverses exactly as expected. LoRA is flawless on dev with a validation
loss of 1.3 x 10^-11, and the CNN-BiLSTM reaches 100% validation accuracy by epoch 4. **Every
model is perfect on the corpus it was fitted to, and the most perfect one transfers worst.**

What pretraining buys is visible here too: the frozen probe keeps 99.82 AUC while its calibration
moves, so its ordering is intact. Both fitted models fall to 93-94 AUC and 13-14% EER, losing
discrimination outright. That is why a threshold can in principle rescue the first and cannot
rescue the other two.

**The fix transfers to all three**, 29.2 to 0.11%, 65.7 to 0.27%, 84.7 to 0.27%. The cost is
stated honestly because it is not free outside the frozen case: the CNN-BiLSTM pays 0.07 to 1.10%
false acceptance and slips from 100% to 99.73% on in-domain dev, and LoRA pays 0.02 to 0.09%. A
frozen representation with a linear head absorbs new speakers at no cost. A model that can move
spends a little capacity on them.

### 6.6 Per speaker, the escalation erases the survivors

Corpus averages hide the shape of this. Median per-speaker false rejection on held-out genuine
speech:

| Architecture | Hindi-English median | Bengali-English median | Speakers below 5% (of 22 / 32) |
|---|---|---|---|
| XLS-R + linear probe | 47.1% | 1.5% | 0 / **17** |
| CNN-BiLSTM | 81.8% | 59.9% | 0 / 0 |
| **XLS-R + LoRA** | **100.0%** | **93.4%** | **0 / 0** |
| any of the three, fixed | 0.0% | 0.0% | 22 / 32 |

The adapted model rejects **every single utterance** from the median Hindi-English speaker. Not
most of them, all of them.

The second column is the one to argue from. Under the frozen probe, 17 of 32 Bengali speakers came
through essentially unharmed. Under either fitted representation that group does not exist, and
nobody in either corpus is below 5%. So the partial survival was not a property of those speakers,
it was a property of a representation that had never been fitted to the training corpus.
**Adapting the model does not shift a distribution, it removes the tail that was working.**

The two corpora also fail in different shapes, which is worth noting because reporting a single
mean for either would describe nobody. Hindi-English is uniform damage, not one of 22 speakers
escapes and the best is still rejected 7.1% of the time. Bengali-English is bimodal, 17 of 32
speakers sit below 5% while a tail runs to 83.6%, so the headline 15.5% describes no actual
speaker.

### 6.7 A published state-of-the-art detector is unusable here

AASIST, released with weights trained by its own authors on ASVspoof2019 LA, run exactly as
released, with only the decision threshold fitted on in-domain dev (the same protocol every other
model here was given):

| Condition | AUC | EER | FAR | FRR |
|---|---|---|---|---|
| in-domain dev, unseen speakers | 46.06 | 48.18 | 48.26 | 48.03 |
| in-domain control, official test partition | 46.11 | 46.40 | 56.65 | 36.45 |
| Hindi-English, all | 43.96 | 53.14 | 52.37 | 53.74 |
| Bengali-English, all | 33.94 | 58.97 | 93.80 | 23.78 |
| out of corpus, pooled | 38.44 | 56.80 | 77.12 | 35.84 |

**It performs at chance, and it does so in domain as well.** AUC 46.06 on the very corpus the
other three detectors solve perfectly.

That first row decides how this result may be used, and it is worth being strict about. AASIST
does not fail on the *target* corpora specifically, it fails on *everything in this dataset*, so
it is **not evidence for architecture independence** and is never presented as such. That claim
rests on the three detectors trained here, which is enough.

What it is instead is the motivation, measured rather than asserted: an off-the-shelf
state-of-the-art detector, applied to narrowband Indian speech against current generators, is
indistinguishable from a coin flip. Two causes are plausible and this design cannot separate them,
so both are stated: every file here is bottlenecked to 4 kHz while ASVspoof2019 is full band, and
XTTS-v2 and FreeVC postdate its training attacks by years.

Two checks were run before believing it. The result is robust to the score convention, since an
in-domain AUC of 46.06 flips to 53.94, chance either way. And because AASIST alone consumes raw
amplitude with no per-utterance normalisation, and generated files here sit systematically louder
than real ones, loudness was a live confound for this model only. Peak-normalising moved the
scores (the two protocols correlate 0.9355) and left it equally broken at AUC 35.33.

### 6.8 In-domain evaluation is saturated to uselessness

A quieter result with a sharp practical consequence. **23 of 25 frontend layers sit at exactly
0.00% dev EER**, including layer 0, which is barely more than a learned filterbank. Dev log-loss
differences are in the fourth decimal.

So the in-domain metric cannot inform any design decision. Choosing a layer by tie-breaking on
that noise picks layer 7, which gives 6.69% out-of-domain EER where layer 20 gives 2.37%. **A
2.8x swing decided by noise.**

Depth buys transfer, and only the out-of-domain number can see it. Hindi-English EER by layer with
an identical head: 26.8% at layer 0, 14.3% at 4, 5.7% at 6, 3.5% at 16, **2.37% at 20**, 7.3% at
24. On Bengali, speakers rejected more than half the time go from 40 of 40 at layer 0 to 4 of 40
at layer 20. Shallow layers hold a corpus-specific cue that is sufficient for perfect in-domain
separation and worthless outside it.

### 6.9 The fix, and how little data it takes

Adding target-corpus speakers to training. Added speakers are held out of every number reported,
five random draws per point, threshold calibrated on in-domain dev exactly as a deployment would
have to:

| speakers added | Hindi-English rejected | Bengali-English rejected |
|---|---|---|
| 0 | 51.7% | 15.3% |
| 1 | 27.5% (σ 17.9) | 9.3% (σ 1.5) |
| 2 | 16.3% (σ 13.7) | 4.5% (σ 4.2) |
| 4 | **0.8%** (σ 0.6) | 3.5% (σ 5.3) |
| 8 | 0.2% | 0.2% |
| 16 | 0.1% | 0.1% |

Four speakers is roughly 400 utterances against a training set of 11,410. **A 7% increase in data
producing a 65x reduction in false rejection.** That is a diversity effect, not a scale effect.

Three things about this curve matter more than its endpoint.

**It costs nothing in domain.** Dev EER stays at 0.00% and control rejection at 0.0% at every
point.

**It transfers across corpora.** Adding *Hindi*-English speakers drops *Bengali*-English rejection
from 15.3% to 0.4%, and adding Bengali speakers drops Hindi-English from 51.7% to 2.5%. The
detector is not learning the one recording chain it was shown, it is learning to tolerate
recording chains. This is the difference between a finding and a patch.

**The variance at one speaker is itself a result.** σ 17.9 on a mean of 27.5 at n=1, falling to
σ 0.6 by n=4. One added speaker sometimes fixes the problem and sometimes does not, so the
requirement is not "some target data" but "enough draws that the outcome no longer depends on
which speaker happened to be drawn".

### 6.10 It survives real phones

Every number above comes from studio-grade read-speech corpora. The deployed model will be handed
a phone, in a room, over a codec, by someone talking rather than reading. That is a third
recording condition, and the entire finding of this project is that an unfamiliar recording
condition breaks detectors. So this was the test most likely to contradict everything else.

45 clips from 26 speakers, recorded on their own phones in their own rooms, through three routes:
the phone's recorder app, WhatsApp voice notes, and call recordings. 4.2 to 63.2 seconds, arriving
at 44.1 and 48 kHz.

**The shipped model rejected none of it.** 0 of 44 in-scope clips, from 25 speakers. The baseline
rejected one, at a score of +0.03, which is the decision boundary to two decimal places.

Margins matter more than the count here. The fixed head's mean score is -9.42 (negative is
genuine), median -9.83, and its closest clip is -2.42, while fakes through the same graph sit at
+8 to +15. The separation is about twenty points wide, so this is not a detector that has quietly
learned to answer "real" to everything.

**The bound is stated honestly: 0 of 44 is not 0%.** The 95% upper bound is about 7% at clip
level, and because rejections cluster by speaker, 0 of 25 speakers puts it nearer 11%. The
supported claim is "no false rejection observed across 25 speakers and three channels", not "zero
false rejection".

One detail from this test sharpened the main claim. The three sources really are three different
channels, not three names for one: median energy above 4 kHz before the bottleneck is 0.0054 for
the phone recorder, 0.0045 for WhatsApp, and **0.0000** for call recordings. Call recordings
arrive natively narrowband, so the bottleneck does nothing to them, and they are the only source
where the baseline failed. What broke it there was the codec, not the bandwidth.

### 6.11 A side investigation: training on resynthesis instead of on generators

This was added mid-project, forced by a real failure rather than planned. A downloaded English
clip from an unknown generator came back **genuine at full margin from both heads**. XTTS-v2 and
FreeVC both decode through HiFi-GAN, so between them the detector had seen exactly one vocoder
family.

The idea, following published work (Wang and Yamagishi, arXiv 2309.06014): instead of supplying
fakes as "audio some TTS produced", supply them as "audio that has been through a resynthesis
bottleneck", which every generator has. Passing a *real* recording through a vocoder isolates
exactly that, because utterance, speaker, words and channel are all held fixed, and the genuine
twin of every fake sits in the same training set. 34,820 fakes were built through four
deliberately dissimilar paths: a neural codec, classical phase reconstruction, linear prediction,
and spectral rank truncation.

**It was parked as a contribution, and that was the right call**, because it does not yield a
deployable detector: every variant that detects well rejects 45-52% of genuine out-of-corpus
speech against the original head's 25%. Three results are kept.

**The motivation, as a number.** A head trained on XTTS-v2 and FreeVC catches only **9-32%** of
speech put through a resynthesis path it never saw. What such a detector holds is one vocoder
family, not synthesis.

**Unseen-attack difficulty is a property of the attack, not a constant.** Holding out each path in
turn: `griffinlim` is caught at **99.1%** while unseen, `encodec` at 55.8%, `lowrank` at 15.3%,
`lpc` at 2.7%. Reporting any single held-out path as *the* unseen-attack rate would have been a
statement about which path happened to be drawn. Also, resynthesis training reaches TTS (XTTS-v2
detected at 65-85% by heads that saw no TTS at all) and does not reach voice conversion (FreeVC at
0.4-9.9% in every variant).

**And it corroborates the main claim from outside it.** Fitting harder to the training corpus cost
genuine speech again, under a completely different training set that the central finding was not
derived from.

## 7. Things that nearly went wrong

These are the confounds that were found and closed. They are listed because in a project whose
entire claim is "the model learned something other than what it was supposed to", the credibility
of the result rests on how hard that possibility was hunted.

**A perfect score deserves suspicion before it deserves belief.** The in-domain baseline hit 0%
EER during feature extraction, which triggered the bandwidth audit above rather than a
celebration.

**Measure both sides as stored files, not as arrays in memory.** After a bottleneck this
aggressive, all that survives above 4 kHz is the 16-bit quantisation floor, which a float32 array
in memory does not have. An early one-sided check read the fake four decades *below* the real and
called it matched. A fake that far below is just as separable as one above.

**A shared offset across unlike methods is the pipeline, not the methods.** All four vocoder paths
came out 3-8x below the real side on energy above 4 kHz. Four unrelated algorithms do not agree by
coincidence, so the agreement itself was the evidence. The cause was one extra resample: the real
corpus passed the bottleneck once, a vocoded file passes it again after resynthesis. Fixed by
giving the genuine class an identical processing history.

**Medians are not evidence of separability.** The first version of that check compared medians on
about ten clips per path and reported a 3-8x gap, which reads as damning and was worth almost
nothing. The same data measured properly gives AUC 0.610, not 0.95. A single-scalar AUC on matched
pairs costs minutes and is the cheapest thing that can be run on any new data before training on
it. After the fix it fell to 0.536.

**A residual confined to one method is signal, and should be left alone.** `griffinlim` did not
move (0.576 to 0.578) because iterative phase reconstruction genuinely alters high-frequency
content. That is an artifact of the resynthesis, which is the thing the detector is supposed to
find. Removing it would have removed the task.

**Verify a control in representation space, not only in the signal.** The passthrough genuine
class was checked by cosine similarity against its own source inside the frozen frontend: median
1.0000, minimum 0.9998, so it is a genuine no-op, while the fakes sit at median 0.9975. A control
that is invisible to a signal-level check but visible to the frontend would have replaced one
confound with another.

**The optimisation that makes extraction fast blunts its own verification.** Batching by duration
minimises padding, and padding is exactly what an attention-mask check needs in order to detect
anything. Re-tested with 90-94% of each batch as padding, alone-versus-padded agreement is
6.7 x 10^-7.

**A proxy that correlates with the grouping variable measures the grouping.** "Contains a short
Latin token" was used to test whether the TTS was dropping English words, but it is very nearly a
test for "is this file code-switched", so it compared conditions rather than English content.
Binning by Latin-script share instead leaves the duration ratio flat at about 0.91 across every
band.

**Level is a confound in the audio that cannot reach the features.** Generated files sit
systematically louder than real ones, and in a five-scalar channel control `log rms` dominates.
XLS-R's feature extractor applies per-utterance normalisation, so absolute level is structurally
invisible to it. Worth stating rather than fixing, and it is exactly why AASIST, which has no such
normalisation, needed its own loudness control.

## 8. Limitations, stated plainly

- **Two language pairs, one corpus family.** The corpus effect is measured between partitions of a
  single collection effort, not across independent collections.
- **Fakes are synthetic by construction**, from two generator families with a third held out. No
  in-the-wild deepfakes are included.
- **Pooled features over a frozen frontend.** Frame-level modelling and frontend adaptation are
  not covered by these results.
- **The switching contrast is about 2x**, not the infinite contrast a pure zero-switch condition
  would give. A null is in principle ambiguous between "no effect" and "not enough contrast", and
  what resolves it here is the monolingual same-corpus bin, which supplies the zero-switch
  condition directly.
- **Rejection rates depend on a threshold that a saturated dev set cannot determine.** EER figures
  are threshold-free. Rejection figures are reported at multiple operating points for this reason.
- **Logical access only. A deepfake played through a loudspeaker and re-recorded on a microphone
  is accepted as genuine, and this is not an incidental gap.** Both heads returned genuine at the
  widest margin the interface can express. The acoustic path smears the synthesis artifacts, and
  more importantly it replaces the synthetic recording chain with a real microphone in a real
  room.
  This is the project's own claim arriving from the other direction. If the decision boundary
  separates recording conditions rather than synthesis, then handing a fake an authentic recording
  condition should defeat it, and it does. The field already treats these as separate problems,
  since ASVspoof runs Logical Access and Physical Access as distinct tracks with distinct
  detectors, and nothing here was trained on replay. The deployed interface warns about this at
  the recorder itself.

## 9. The demo, from upload to verdict

A clip is scored twice, once by the baseline detector and once by the fixed one, so the whole
argument of the project is visible on a single upload.

Backend: `https://aryankumarpandey05--codeswitch-deepfake-api-api.modal.run`
Frontend: `codeswitch-deepfake.vercel.app`

### What happens to an uploaded clip

1. **Decode and resample.** The file is converted to 16 kHz mono and pushed through the same
   native to 8 kHz to 16 kHz bottleneck every training file passed through. This is not optional.
   These weights have never seen energy above 4 kHz, so skipping it puts the input out of
   distribution in precisely the way this project is about.
2. **Window it.** Pooled features make clip length part of the method: a mean and standard
   deviation over frames is not length-invariant, and training utterances average 4.7 seconds, so
   a 30 second clip pooled whole is a different statistic rather than a longer one. The input is
   cut into 5 second spans at a 4 second hop, spans quieter than a tenth of the clip's own level
   are dropped (each span is normalised on its own, so silence would be amplified into noise), and
   the per-span logits are averaged. Validated by gluing utterances with known verdicts into
   examiner-length clips and rescoring: on 19 to 49 second clips, windowed drift is 0.53 against
   whole-clip 2.34 on genuine speech, at 100% verdict agreement.
3. **Run the frontend.** XLS-R truncated to 20 transformer blocks, dynamically quantised to int8,
   as an ONNX graph on CPU.
4. **Score both heads.** Two logistic heads over the 2,048-dimensional pooled output. The decision
   threshold is folded into each head's bias, so "probability above one half means fake" is the
   calibrated rule rather than an accident of where logistic regression puts its own zero.
5. **Return two verdicts**, a per-window timeline, and the metadata, so the difference between the
   two heads is inspectable rather than asserted.

### What the deployment cost, measured rather than assumed

**Quantisation error does not reach the decision, but it is not uniform across models.** Dynamic
int8 over MatMul costs 1.5% relative error on the pooled features. On the fixed head that is 0.0%
verdict flips at a score correlation of 0.9989. On the baseline, which sits near its own boundary
on out-of-domain genuine speech, the same error moves rejection from 45.0% to 41.7%. The general
point is that a quantisation budget depends on the margin, not on the model.

**A deployable model is not the same as a fast one.** The truncated int8 graph needs about 430 ms
of compute per second of audio per core, so a 30 second clip takes 17.8 seconds single-threaded.
Papers reporting EER on a frozen self-supervised frontend rarely state this, and it is the
difference between a result and a demonstration.

**Why int8 ONNX and not PyTorch.** Hosting stays free only if the always-warm container fits in
about 2 GiB. A quantised XLS-R-300M lands around 800 MB, fp32 PyTorch is about 2.5 GB and does not
fit. This is an export-step constraint only, and training was unaffected by it.

## 10. Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Dataset build, fake generation, training and evaluation. One file per stage, numbered in execution order |
| `app/api/` | FastAPI inference service, int8 ONNX under onnxruntime |
| `app/web/` | React frontend, Vite |
| `models/` | The trained classifier heads, the quantised frontend, and a model card |
| `DATASET_CARD_fakes.md` | Provenance, attack systems, controls and known characteristics of the generated half |

The quantised XLS-R frontend is 291 MB and is not in the repository. It is rebuilt by the export
notebook and served from a Modal volume.

## 11. The notebooks, in order

Each file is self-contained, opens with a Requirements block naming its accelerator, its attached
inputs and what it writes, and answers one question.

| Notebook | What it decides or produces |
|---|---|
| `02_build_kaggle_dataset.py` | Downloads, parses, splits, cuts and resamples the real corpora once, and freezes the result as a mountable dataset so no later session ever rebuilds it |
| `03_xtts_gate.py` | Tests whether XTTS-v2 can speak one sentence containing both Devanagari and Latin script and pronounce the English as English. Decides how every fake is generated |
| `04_english_control.py` | Attempts a monolingual Indian English control. Abandoned: the all-English segments of this corpus are UI labels and dictated syntax, not English speech |
| `05_xlsr_frontend_check.py` | Verifies the frontend loads and runs, and weighs the deployed artifact, since free hosting is an arithmetic claim until something measures it |
| `06_attack2_gate.py` | Chooses the second attack system by architecture rather than by weights, and verifies the fakes match the real audio's band |
| `07_generate_fakes.py` | Generates one fake per real utterance under the two matching rules |
| `08_audition_fakes.py` | Establishes that the TTS is not silently dropping English words, which would have made the code-switched fakes monolingual under a code-switched label |
| `09_extract_features.py` | Caches pooled features for all 25 layers, once, so every later experiment is cheap |
| `10_baseline_and_gonogo.py` | Trains the monolingual Hindi baseline, runs the bandwidth control, and measures the collapse |
| `11_verify_cache.py` | Re-tests the attention mask with padding deliberately dominating the batch, because the fast extraction path had blunted its own check |
| `12_decompose_collapse.py` | Attributes the collapse to a cause before acting on it. This is where code-switching is ruled out |
| `13_fix_data_ceiling.py` | Measures how many target speakers it takes to close the gap, and whether the fix transfers between corpora |
| `14_export_models.py` | Exports both detectors along the exact deployment path and reports what quantisation and the runtime cost |
| `15_realvoice_test.py` | Scores real phone recordings locally, so none of that audio leaves the machine |
| `16_metrics_table.py` | Computes accuracy, precision, recall, F1, AUC, EER and both error rates across every condition, once, from the shipped weights |
| `16b_threshold_oracle.py` | Tests whether a better threshold would fix it, by handing the objection every advantage |
| `17_cnn_bilstm.py` | Retrains the pre-self-supervised standard architecture under identical controls |
| `18_lora_finetune.py` | Lets the encoder itself adapt, which separates the finding from the fact of freezing |
| `19_aasist_zeroshot.py` | Runs a published detector that had nothing to do with this setup, exactly as released |
| `20_arch_comparison.py` | Merges every score file under one metric definition, so the headline table is computed once rather than assembled by hand |
| `21_vocoded_fakes.py` | Resynthesises real speech through four unlike vocoders, to teach synthesis rather than a generator |
| `21b_confound_check.py` | Finds and closes the extra-resample confound in that set |
| `22_vocoded_features.py` | Caches the vocoded set into the same feature space as the original cache |
| `22b_vocoded_heads.py` | Trains on resynthesis and measures what it buys and what it costs |

## 12. Reproducing this

The notebooks run top to bottom on Kaggle and are numbered in execution order. `KAGGLE_USER` at
the top of each file is the dataset owner and is the only line to change when running under
another account.

Training used Kaggle's free tier throughout, two T4 GPUs, well inside the weekly quota. The
laptop-only steps (the metrics table, the threshold oracle, the architecture merge) run in seconds
with numpy and pandas alone.

Both datasets, real and generated, are published on Kaggle and mount read-only.

## 13. Data, licences and ethics

Real speech is cut from MUCS / OpenSLR [103](https://openslr.org/103/) and
[104](https://openslr.org/104/), CC BY-SA 4.0.

Synthetic speech is generated from those same speakers. XTTS-v2 is released under the Coqui Public
Model Licence, which is non-commercial and within scope for academic research. FreeVC is MIT, and
its checkpoint is trained on VCTK (CC BY 4.0), which carries an attribution requirement. Generated
audio inherits those terms. Neither generator is shipped in the deployed demo, which runs the
detector only.

The generated half consists of voice clones of identifiable speakers who consented to an ASR
corpus release. Publishing synthetic speech derived from a public research corpus, under that
corpus's licence and with an ethics statement, is the established norm in this field rather than
an exception: ASVspoof's Logical Access sets are spoofed speech generated from VCTK, WaveFake
publishes generated audio derived from LJSpeech, and MLAAD and IndicSynth do the same over public
corpora.

The real-phone acceptance test is the one place where audio from people outside any corpus was
used. It was run entirely on a local machine, nothing was uploaded to any third party, and the
only artifact written was a CSV of scores.
