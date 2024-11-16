from transformers import Trainer
from sklearn.metrics import classification_report, accuracy_score
from metrics.evaluate_model import calculate_metrics

class EvaluationPipeline:
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset

    def evaluate(self):

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        predictions = trainer.predict(self.test_dataset)
        preds = predictions.predictions.argmax(axis=1)
        labels = predictions.label_ids
        
        metrics = calculate_metrics(preds, labels)
        metrics["Accuracy"] = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=self.test_dataset.features["label"].names)
        
        return {
            "metrics": metrics,
            "classification_report": report,
        }