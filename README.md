# CodeSwitch-Deepfake

**Audio deepfake detection for code-switched Indian speech (Hindi-English and Bengali-English),
and a measurement of why these detectors fail here.**

Deepfake voice detectors do break on Indian speech, but not for the reason the field assumed.
They do not break because of the language, and they do not break by letting fakes through. They
break by rejecting real human beings, roughly half of them, because they learned what the
training microphone sounded like instead of what synthetic speech sounds like. Adding eight
speakers from a new recording setup takes that error from 29-85% down to 0.1-0.3%.

**[Live demo](https://codeswitch-deepfake.vercel.app)** · one upload, scored twice, so the
difference between the baseline and the fixed model is visible on a single clip.

**[Full documentation](DOCUMENTATION.md)** covers every dataset decision, all eleven
experiments, the deployment, and the near-misses. This file is the summary.

## The finding

Frozen XLS-R probe, trained only on monolingual Hindi. Genuine speech rejected, at the deployed
threshold:

| Test condition | Genuine speech rejected |
|---|---|
| Monolingual Hindi, training corpus, speakers never seen before | **0.0%** |
| Monolingual Hindi, the corpus's official test partition | **0.0%** |
| Monolingual Hindi, recorded inside the Hindi-English corpus | **51.5%** |
| Code-switched Hindi-English | **49.2%** |
| Monolingual Bengali, recorded inside the Bengali-English corpus | **23.0%** |
| Code-switched Bengali-English | **14.4%** |

Read rows 1 and 3 together. Same language, same monolingual content, zero code-switching, unseen
speakers in both. The only thing that changed is which recording session the audio came from, and
rejection goes from 0.0% to 51.5%.

Now read rows 3 and 4. Monolingual speech is rejected slightly *more* than the code-switched
speech sitting beside it in the same corpus, and the same holds in Bengali, 23.0% against 14.4%.
Switching is not the axis.

Pooled across every out-of-corpus condition tested:

| | |
|---|---|
| Fakes missed (false acceptance) | **0.00%** |
| Genuine speech rejected (false rejection) | **29.2%** |

**Every single fake was caught. Nearly a third of real people were not.** The boundary this
detector learned is not "synthetic versus genuine", it is "matches my training channel versus
does not".

It is not an artifact of one architecture. Across four detectors, rejection tracks how much
capacity was fitted to the training corpus: 29.2% for the frozen probe, 65.7% for CNN-BiLSTM,
84.7% for LoRA. Fine-tuning on a single corpus, the field's default recipe, nearly triples it. Refitting the threshold on the entire target corpus still leaves 44.5% and 18.4%, so
recalibration is not the answer.

**The fix is data, and it is small.** Eight speakers per corpus, held out of every reported
number, take all three trained detectors to 0.11-0.27%. Four speakers were already enough, and it
transfers to a language pair it was not fitted on. It is free for the frozen probe; the two
fitted representations pay a little false acceptance for it, and that cost is stated in the
documentation rather than rounded away.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Dataset build, fake generation, training and evaluation. One file per stage, numbered in execution order |
| `app/api/` | FastAPI inference service, int8 ONNX under onnxruntime |
| `app/web/` | React frontend, Vite |
| `models/` | The trained classifier heads, the quantised frontend, and a model card |
| `DATASET_CARD_fakes.md` | Provenance, attack systems, controls and known characteristics of the generated half |
| `DOCUMENTATION.md` | The full technical account |

The quantised XLS-R frontend is 291 MB and is not in the repository. It is rebuilt by the export
notebook and served from a Modal volume.

## Reproducing this

The notebooks run top to bottom on Kaggle and are numbered in execution order. `KAGGLE_USER` at
the top of each file is the dataset owner and is the only line to change when running under
another account. Each one opens with a Requirements block naming its accelerator, its attached
inputs and what it writes. [`DOCUMENTATION.md` §11](DOCUMENTATION.md#11-the-notebooks-in-order)
says what each decides.

Training used Kaggle's free tier throughout, two T4 GPUs, well inside the weekly quota. The
laptop-only steps run in seconds with numpy and pandas alone. Both datasets, real and generated,
are published on Kaggle and mount read-only.

## Limitations

Stated here in short; [`DOCUMENTATION.md` §8](DOCUMENTATION.md#8-limitations-stated-plainly)
gives each one in full.

- Two language pairs from one corpus family. The corpus effect is measured between partitions of
  one collection effort, not across independent collections.
- Fakes are synthetic by construction, from two generator families with a third held out. No
  in-the-wild deepfakes.
- **Logical access only.** A deepfake replayed through a loudspeaker and re-recorded on a
  microphone is accepted as genuine. This is the project's own claim arriving from the other
  direction, since handing a fake an authentic recording condition defeats a detector that
  separates recording conditions. The deployed interface warns at the recorder itself.
- Rejection rates depend on a threshold a saturated dev set cannot determine. EER figures are
  threshold-free; rejection figures are reported at multiple operating points for that reason.

## Data, licences and ethics

Real speech is cut from MUCS / OpenSLR [103](https://openslr.org/103/) and
[104](https://openslr.org/104/), CC BY-SA 4.0.

Synthetic speech is generated from those same speakers. XTTS-v2 is released under the Coqui
Public Model Licence, which is non-commercial and within scope for academic research. FreeVC is
MIT, and its checkpoint is trained on VCTK (CC BY 4.0), which carries an attribution requirement.
Generated audio inherits those terms. Neither generator is shipped in the deployed demo, which
runs the detector only.

The generated half consists of voice clones of identifiable speakers who consented to an ASR
corpus release. Publishing synthetic speech derived from a public research corpus, under that
corpus's licence and with an ethics statement, is the established norm in this field rather than
an exception: ASVspoof's Logical Access sets are spoofed speech generated from VCTK, WaveFake
publishes generated audio derived from LJSpeech, and MLAAD and IndicSynth do the same over public
corpora.

The real-phone acceptance test is the one place where audio from people outside any corpus was
used. It was run entirely on a local machine, nothing was uploaded to any third party, and the
only artifact written was a CSV of scores.
