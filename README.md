# CodeSwitch-Deepfake

Audio deepfake detection for code-switched Indian speech, Hindi-English and Bengali-English.

The detectors in this repository catch every synthetic clip they are shown. The problem is
what they do to genuine speech recorded somewhere else: out of corpus, a frozen XLS-R probe
rejects 29.2% of real human speech, a CNN-BiLSTM rejects 65.7%, and the same XLS-R adapted
with LoRA rejects 84.7%, while false acceptance never rises above 0.07%. Rejection tracks how
much of the model was fitted to the training corpus, so fine-tuning a self-supervised frontend
on a single corpus, which is the usual recipe, nearly triples false rejection against leaving
it frozen.

Language is not the cause. Monolingual Hindi drawn from the code-switched corpus is rejected
at 51.5%, slightly more than the code-switched speech beside it at 49.2%, while the same
language from the training corpus is rejected at 0.0%. What moves the number is the recording
condition, not the switching.

Adding eight speakers per target corpus to training closes it: 29.2% to 0.11%, 65.7% to 0.27%,
84.7% to 0.27%, at no in-domain cost for the frozen probe. Four speakers is already most of
the effect, and it transfers across languages, so adding Hindi-English speakers also repairs
Bengali-English.

## Demo

A clip is scored twice, once by the baseline detector and once by the fixed one, so the
difference is visible on a single upload.

Backend: `https://aryankumarpandey05--codeswitch-deepfake-api-api.modal.run`

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | Dataset build, fake generation, training and evaluation, one file per stage |
| `app/api/` | FastAPI inference service, int8 ONNX under onnxruntime |
| `app/web/` | React frontend, Vite |
| `models/` | The trained classifier heads and a model card |

The quantised XLS-R frontend is 291 MB and is not in the repository. It is rebuilt by the
export notebook and served from a Modal volume.

## Data

Real speech is cut from MUCS / OpenSLR [103](https://openslr.org/103/) and
[104](https://openslr.org/104/), CC BY-SA 4.0. Every file is resampled native to 8 kHz to
16 kHz on purpose: the monolingual sets are natively 8 kHz and the code-switched sets 16 kHz,
and that split falls on the train/test boundary, so without the bottleneck any result here
would be attributable to bandwidth rather than to the corpus.

Synthetic speech is generated from those same speakers with XTTS-v2 and FreeVC, one fake per
real utterance. A third system is held out entirely so attack generalisation can be measured
against data no training run has seen. See `DATASET_CARD_fakes.md`.

Both datasets are published on Kaggle.

## Reproducing

The notebooks run top to bottom on Kaggle and are numbered in execution order. Each one opens
with a Requirements block naming its accelerator, its attached inputs and what it writes.
`KAGGLE_USER` at the top of each file is the dataset owner and is the only line to change when
running under another account.

Training used Kaggle's free tier throughout, two T4s, well inside the weekly quota.

## Licence and attribution

Source corpora are CC BY-SA 4.0 and are attributed above. XTTS-v2 is released under the Coqui
Public Model Licence, which is non-commercial; FreeVC carries VCTK's CC BY 4.0 attribution
requirement. Generated audio inherits those terms.
