import os
from transformers import pipeline

file_dir = os.path.dirname(os.path.realpath(__file__))

model_path = f'{file_dir}/models/finetuned-1e-5/checkpoint-2000'
generator = pipeline('text-generation', model=model_path, device=0)
out = generator("And it", max_length=200)[0]['generated_text']
print('\n'.join(out.splitlines()))
