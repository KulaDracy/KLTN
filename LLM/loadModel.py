from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Chọn model (có thể thay bằng mistral-7b-instruct hoặc llama-2-7b-chat)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Tạo pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test: tóm tắt nội dung
prompt = """
Bạn là một trợ lý AI. Nhiệm vụ: tóm tắt đoạn văn sau thành 3 gạch đầu dòng cho slide thuyết trình:

"Deep learning models have achieved remarkable success in computer vision and NLP.
However, training requires massive labeled data, which is costly and time-consuming.
Synthetic data generation offers an alternative, reducing costs and improving scalability."
"""

output = llm(prompt, max_new_tokens=200, temperature=0.7)
print(output[0]["generated_text"])
