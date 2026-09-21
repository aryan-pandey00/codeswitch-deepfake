# codeswitch-deepfake-models

Artifacts served by `app/api/service.py`.

* `frontend_int8.onnx`, `facebook/wav2vec2-xls-r-300m` truncated to 20 transformer blocks, dynamically
  quantised to int8 over MatMul. One input, `input_values`, shape (1, samples) at 16000 Hz,
  already normalised to zero mean and unit variance. One output, `last_hidden_state`, which is
  hidden state 20 of the untruncated model.
* `heads.json`. Two logistic heads over the mean and standard deviation of that output,
  2048 dimensions. Each carries `mu`, `sd`, `w`, `b`. The decision threshold is folded
  into `b`, so the service's rule of "probability above a half means fake" is the calibrated
  rule and not an accident of where logistic regression puts its own zero.

## Length

Scores are pooled over frames, and the training utterances average under five seconds, so a
long upload is cut into 5 second spans at a 4 second hop and the spans
are averaged as logits. Spans quieter than a tenth of the clip's own level are dropped, since
each span is normalised on its own and silence would otherwise be amplified into noise.

## Preprocessing that is not optional

Every training file was resampled native to 8 kHz to 16 kHz, so no energy above 4 kHz has ever
been seen by these weights. An upload arriving at 44.1 kHz must pass the same bottleneck. This
is the exact axis the project measured, and skipping it puts the input out of distribution in
the way the paper is about.

## What the two heads are

`baseline` is trained on one corpus. It reaches 0% error in domain and then rejects a large
fraction of genuine speech recorded anywhere else. It is the exhibit, not the product.

`fixed` adds 8 speakers per code-switched corpus to the same training set. Those speakers
are held out of every number reported, so the figures describe these weights rather than a
sibling.

## Known limits

Neither head has heard a phone microphone. Both corpora used for adaptation are read speech
from studio-grade collections, so a live recording is a third recording condition and not a
familiar one. The acceptance test on real phone audio is a separate step and its result is not
predicted by anything here.
