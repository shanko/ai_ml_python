from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'google/datagemma-rag-27b-it'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)

input_text = """Your role is that of a Question Generator.  Given Query below, come up with a
maximum of 25 Statistical Questions that help in answering Query.

These are the only forms of Statistical Questions you can generate:
1. What is $METRIC in $PLACE?
2. What is $METRIC in $PLACE $PLACE_TYPE?
3. How has $METRIC changed over time in $PLACE $PLACE_TYPE?

where,
- $METRIC should be a metric on societal topics like demographics, economy, health,
  education, environment, etc.  Examples are unemployment rate and
  life expectancy.
- $PLACE is the name of a place like California, World, Chennai, etc.
- $PLACE_TYPE is an immediate child type within $PLACE, like counties, states,
  districts, etc.

Your response should only have questions, one per line, without any numbering
or bullet.

If you cannot come up with Statistical Questions to ask for a Query, return an
empty response.

Query: What are some interesting trends in Sunnyvale spanning gender, age, race, immigration, health conditions, economic conditions, crime and education?
Statistical Questions:"""
inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

outputs = model.generate(**inputs, max_new_tokens=4096)
answer = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
print(answer)

