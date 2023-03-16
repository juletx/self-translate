import csv
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

langs_xstory = ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]

lang_names = [
    "Russian",
    "Chinese",
    "Spanish",
    "Arabic",
    "Hindi",
    "Indonesian",
    "Telugu",
    "Swahili",
    "Basque",
    "Burmese",
]


def load_model(model_name):
    """Load a model and tokenizer.

    Args:
        model_name (string): model name

    Returns:
        tokenizer, model: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.cuda()
    return tokenizer, model


def get_dataset():
    """Load the xStoryCloze dataset.

    Returns:
        xstory_cloze: dataset
    """
    xstory_cloze = {}
    for lang in langs_xstory:
        xstory_cloze[lang] = load_dataset("juletxara/xstory_cloze", lang)
    return xstory_cloze


def translate_example(example, tokenizer, model, prompt):
    """Translate an example.

    Args:
        example (dict): example
        tokenizer (tokenizer): tokenizer
        model (model): model
        prompt (string): prompt

    Returns:
        translated_example: translated example
    """
    sentences = [
        f"""{prompt} {example["input_sentence_1"]}\nEnglish:""",
        f"""{prompt} {example["input_sentence_2"]}\nEnglish:""",
        f"""{prompt} {example["input_sentence_3"]}\nEnglish:""",
        f"""{prompt} {example["input_sentence_4"]}\nEnglish:""",
        f"""{prompt} {example["sentence_quiz1"]}\nEnglish:""",
        f"""{prompt} {example["sentence_quiz2"]}\nEnglish:""",
    ]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to("cuda")
    translated_tokens = model.generate(
        **inputs,
        max_length=50,
    )
    translated_sentences = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )
    # remove prompt from generation results
    translated_example = {
        "story_id": example["story_id"],
        "input_sentence_1": translated_sentences[0][len(sentences[0]) :],
        "input_sentence_2": translated_sentences[1][len(sentences[1]) :],
        "input_sentence_3": translated_sentences[2][len(sentences[2]) :],
        "input_sentence_4": translated_sentences[3][len(sentences[3]) :],
        "sentence_quiz1": translated_sentences[4][len(sentences[4]) :],
        "sentence_quiz2": translated_sentences[5][len(sentences[5]) :],
        "answer_right_ending": example["answer_right_ending"],
    }
    del inputs, translated_sentences
    torch.cuda.empty_cache()
    return translated_example


def save_file(translated_examples, lang, split, name):
    """Save the translated dataset to a tsv file.

    Args:
        translated_examples (list): list of translated examples
        lang (string): language
        split (string): train or eval
        name (string): model name
    """
    dirname = f"../datasets/xstory_cloze_mt/{name}"
    filename = f"{dirname}/spring2016.val.{lang}.tsv.split_20_80_{split}.tsv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "story_id",
                "input_sentence_1",
                "input_sentence_2",
                "input_sentence_3",
                "input_sentence_4",
                "sentence_quiz1",
                "sentence_quiz2",
                "answer_right_ending",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for example in translated_examples:
            writer.writerow(example)


def get_prompt_few_shot(xstory_cloze, lang, lang_name, shots):
    """Get the prompt for few-shot translation.

    Args:
        xstory_cloze (dict): dataset
        lang (string): language
        lang_name (string): language name
    """
    prompt = ""
    for i in range(shots):
        prompt += f"""{lang_name}: {xstory_cloze[lang]["train"][i]["input_sentence_1"]}\n"""
        prompt += f"""English: {xstory_cloze["en"]["train"][i]["input_sentence_1"]}\n"""
    prompt += f"""{lang_name}: """
    return prompt


def translate_dataset(xstory_cloze, tokenizer, model, model_name):
    """Translate the dataset.

    Args:
        xstory_cloze (dict): dataset
        model (model): model
        model_name (string): model name
    """
    name = model_name.split("/")[-1]
    for i, lang in enumerate(langs_xstory):
        prompt = get_prompt_few_shot(xstory_cloze, lang, lang_names[i], 4)
        translated_examples = []
        for example in tqdm(
            xstory_cloze[lang]["eval"],
            total=len(xstory_cloze[lang]["eval"]),
            desc=f"Translating {lang}",
        ):
            translated_example = translate_example(example, tokenizer, model, prompt)
            translated_examples.append(translated_example)
        save_file(translated_examples, lang, "eval", name)


def main():
    """Main function."""
    parser = ArgumentParser("Evaluate a model on the xStoryCloze dataset")
    parser.add_argument(
        "model_name",
        type=str,
        help="Huggingface model name or path to a local model",
    )
    args = parser.parse_args()

    model_name = args.model_name
    xstory_cloze = get_dataset()
    tokenizer, model = load_model(model_name)
    translate_dataset(xstory_cloze, tokenizer, model, model_name)


if __name__ == "__main__":
    main()
