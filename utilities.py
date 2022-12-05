import openai
from transformers import pipeline, set_seed
import torch
from tqdm import tqdm


def get_gpt3_outputs(prompts, model="text-davinci-003", max_requests=20, max_tokens=25, temperature=0.7):
    """
    Get the outputs from GPT-3 for a list of prompts.
    """
    all_outputs = []
    i = 0
    while i < len(prompts):
        completions = openai.Completion.create(
            model=model,
            prompt=prompts[i:min(i + max_requests, len(prompts))],
            max_tokens=max_tokens,
            temperature=temperature
        )["choices"]
        for completion in completions:
            all_outputs.append(completion["text"].lower())
        i += max_requests
    return all_outputs


def get_gpt2_outputs(prompts, max_tokens=25, temperature=0.0001):
    """
    Get the outputs from GPT-2 for a list of prompts.
    """
    generator = pipeline('text-generation', model='gpt2', device=0)
    outputs = generator(prompts, max_length=max(len(prompt) for prompt in prompts) + max_tokens,
                        temperature=temperature)
    processed_outputs = []
    for output, prompt in zip(outputs, prompts):
        processed_outputs.append(output[0]["generated_text"][len(prompt):].lower())
    return processed_outputs


def get_t0_outputs(prompts, tokenizer, model, batch_size=1, max_tokens=25):
    """
    Get the outputs from T0 for a list of prompts. Optional batching, but this didn't perform well on my system.
    """
    i = 0
    all_outputs = []
    with tqdm(total=len(prompts) / batch_size) as pbar:
        while i < len(prompts):
            tokens = tokenizer.batch_encode_plus(prompts[i:min(i + batch_size, len(prompts))], padding=True)
            outputs = model.generate(torch.tensor(tokens["input_ids"]),
                                     attention_mask=torch.tensor(tokens["attention_mask"]), max_new_tokens=max_tokens)
            decoded = tokenizer.batch_decode(outputs)
            i += batch_size
            all_outputs += decoded
            pbar.update(1)
    return all_outputs


def get_gptneo_outputs(prompts, tokenizer, model, max_tokens=25, temperature=0.01):
    all_outputs = []
    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(
            input_ids.cuda(),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        gen_text = tokenizer.decode(gen_tokens[0])[len(prompt):]
        all_outputs.append(gen_text)
    return all_outputs
