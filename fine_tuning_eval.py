from transformers import AutoTokenizer, AutoModelForSequenceClassification

#使用微调的T5模型
tokenizer = AutoTokenizer.from_pretrained("google/t5-large-crag-evaluator")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/t5-large-crag-evaluator",
    num_labels=3  # Correct, Incorrect, Ambiguous
)

def evaluate_document(query, document):
    """评估文档相关性"""
    input_text = f"Query: {query}\nDocument: {document}\nRelevance:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512)
    outputs = model(**inputs)
    # 获取预测结果和置信度
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs).item()
    confidence = probs[0][predicted_class].item()
    return {
        "class": ["correct", "incorrect", "ambiguous"][predicted_class],
        "confidence": confidence
    }
