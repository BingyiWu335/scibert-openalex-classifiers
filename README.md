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
import torch
from models import load_model

model, tokenizer, labels = load_model("field", "weights/field.pth")
inputs = tokenizer("Research title; Research abstract.",
                   max_length=512, truncation=True, return_tensors="pt")
with torch.inference_mode():
    scores = model(**inputs).softmax(dim=-1)
print(labels[scores[0].argmax().item()])
```

Use `subfield_single` or `subfield_multilabel` for the other checkpoints. For the multilabel model, tokenize the title and abstract as a text pair and apply `sigmoid()` to the logits. Its scores are not normalized to sum to one; the validation-selected threshold is 0.25. The two subfield models have different label orders; use the matching file in `labels/`.

The loader preserves the residual classification heads. No pretrained-weight download is needed. Training and evaluation logs are in `logs/`.

## Citation

If you use these models, please cite the original SciBERT paper.

Beltagy, I., Lo, K. & Cohan, A. SciBERT: A Pretrained Language Model for Scientific Text. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 3615–3620 (ACL, 2019). [ACL Anthology](https://aclanthology.org/D19-1371/)
