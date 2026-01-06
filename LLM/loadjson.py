import json

prompt = """
Tạo dàn ý slide dưới dạng JSON với format:
[
  {"title": "Slide 1", "points": ["...", "..."]},
  {"title": "Slide 2", "points": ["...", "..."]}
]

Nội dung: "Variational Autoencoders are generative models that learn a latent representation of data."
"""

result = llm(prompt, max_new_tokens=300)
print(json.loads(result[0]["generated_text"]))
