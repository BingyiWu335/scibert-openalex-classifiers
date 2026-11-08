# SciBERT OpenAlex classifiers

Fine-tuned SciBERT weights for classifying English research titles and abstracts into OpenAlex fields and subfields.

| Model | Labels | Output | Weights |
|---|---:|---|---|
| Field | 26 | Softmax | [field.pth](https://github.com/BingyiWu335/scibert-openalex-classifiers/releases/download/v1.0/field.pth) |
| Subfield, single-label | 252 | Softmax | [subfield_single.pth](https://github.com/BingyiWu335/scibert-openalex-classifiers/releases/download/v1.0/subfield_single.pth) |
| Subfield, multilabel | 252 | Sigmoid scores | [subfield_multilabel.pt](https://github.com/BingyiWu335/scibert-openalex-classifiers/releases/download/v1.0/subfield_multilabel.pt) |

All three include the encoder and classification head. Base model: [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased). Maximum input length: 512 tokens.

## Use

Download the required checkpoint into `weights/`, then install the dependencies:

```sh
pip install -r requirements.txt
```

```python
from models import load_model, predict

model, tokenizer, labels = load_model("field", "weights/field.pth")
scores = predict(model, tokenizer, ["Research title"], ["Research abstract."])
print(labels[scores[0].argmax().item()])
```

Use `subfield_single` or `subfield_multilabel` for the other checkpoints. The two subfield models have different label orders; use the matching file in `labels/`. Multilabel scores are not normalized to sum to one; the validation-selected threshold is 0.25.

The loader preserves the residual classification heads and input formats. No pretrained-weight download is needed. Checksums are in `SHA256SUMS`; available training and evaluation outputs are in `logs/`.
