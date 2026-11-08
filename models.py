from pathlib import Path
import json

import torch
from torch import nn
from transformers import BertConfig, BertModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
KINDS = {"field": 26, "subfield_single": 252, "subfield_multilabel": 252}


class SingleLabelClassifier(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.scibert = BertModel(config)
        self.block = nn.Sequential(
            nn.Linear(768, 2048), nn.LayerNorm(2048), nn.ReLU(),
            nn.Linear(2048, 768), nn.LayerNorm(768),
        )
        self.classification = nn.Sequential(nn.ReLU(), nn.Linear(768, num_labels))

    def forward(self, **inputs):
        cls = self.scibert(**inputs).last_hidden_state[:, 0]
        return self.classification(self.block(cls) + cls)


class MultilabelClassifier(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.encoder = BertModel(config)
        self.residual_head = nn.Sequential(
            nn.Linear(768, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2048, 768), nn.LayerNorm(768),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, **inputs):
        cls = self.encoder(**inputs).last_hidden_state[:, 0]
        hidden = self.residual_head(cls) + cls
        return self.classifier(self.dropout(self.activation(hidden)))


def load_model(kind, checkpoint, device="cpu"):
    """Load a complete classifier and its ordered OpenAlex labels."""
    if kind not in KINDS:
        raise ValueError(f"Choose one of: {', '.join(KINDS)}")
    config = BertConfig.from_json_file(str(ROOT / "encoder_config.json"))
    cls = MultilabelClassifier if kind == "subfield_multilabel" else SingleLabelClassifier
    model = cls(config, KINDS[kind])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.kind = kind
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizer", local_files_only=True)
    labels = json.loads((ROOT / "labels" / f"{kind}.json").read_text(encoding="utf-8"))
    return model, tokenizer, labels


@torch.inference_mode()
def predict(model, tokenizer, titles, abstracts):
    """Return one score vector per title–abstract pair."""
    if len(titles) != len(abstracts):
        raise ValueError("titles and abstracts must have the same length")
    if not titles:
        return torch.empty((0, KINDS[model.kind]))
    options = dict(padding=True, truncation=True, max_length=512, return_tensors="pt")
    if model.kind == "subfield_multilabel":
        inputs = tokenizer(titles, abstracts, **options)
    else:
        texts = [" ".join(f"{title}; {abstract}".split()) for title, abstract in zip(titles, abstracts)]
        inputs = tokenizer(texts, **options)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = model(**inputs)
    scores = logits.sigmoid() if model.kind == "subfield_multilabel" else logits.softmax(dim=-1)
    return scores.cpu()
