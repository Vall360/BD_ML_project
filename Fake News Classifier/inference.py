import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FakeNewsPredictor:
    def __init__(self, model_dir=None, device=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'Results', 'model')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df: pd.DataFrame, text_column: str, batch_size: int = 32) -> pd.Series:
        texts = df[text_column].fillna('').tolist()
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                results.extend(preds)
        return pd.Series(results, index=df.index, name='prediction')
