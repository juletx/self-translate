from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import csv
import torch
from argparse import ArgumentParser
import os

langs_xstory = ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]
# BCP-47 language codes for the languages in the xStoryCloze dataset
langs_xstory_bcp47 = [
    "rus_Cyrl",
    "zho_Hans",
    "spa_Latn",
    "arb_Arab",
    "hin_Deva",
    "ind_Latn",
    "tel_Telu",
    "swh_Latn",
    "eus_Latn",
    "mya_Mymr",
]


def load_model(model_name):
    """Load a model.

    Args:
        model_name (string): model name

    Returns:
        model: tokenizer and model
    """
    if model_name == "facebook/nllb-moe-54b":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.cuda()
    return model


def get_dataset():
    """Load the xStoryCloze dataset.

    Returns:
        xstory_cloze: dataset
    """
    xstory_cloze = {}
    for lang in langs_xstory:
        xstory_cloze[lang] = load_dataset("juletxara/xstory_cloze", lang)
    return xstory_cloze


def translate_example(example, tokenizer, model):
    """Translate an example.

    Args:
        example (dict): example
        tokenizer (tokenizer): tokenizer
        model (model): model

    Returns:
        translated_example: translated example
    """
    sentences = [
        example["input_sentence_1"],
        example["input_sentence_2"],
        example["input_sentence_3"],
        example["input_sentence_4"],
        example["sentence_quiz1"],
        example["sentence_quiz2"],
    ]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to("cuda")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=200,
    )
    translated_sentences = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )
    translated_example = {
        "story_id": example["story_id"],
        "input_sentence_1": translated_sentences[0],
        "input_sentence_2": translated_sentences[1],
        "input_sentence_3": translated_sentences[2],
        "input_sentence_4": translated_sentences[3],
        "sentence_quiz1": translated_sentences[4],
        "sentence_quiz2": translated_sentences[5],
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
    dirname = f"../datasets/xstory_cloze/{name}"
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


def translate_dataset(xstory_cloze, model, model_name):
    """Translate the dataset.

    Args:
        xstory_cloze (dict): dataset
        model (model): model
        model_name (string): model name
    """
    name = model_name.split("/")[-1]
    for i, lang in enumerate(langs_xstory):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, src_lang=langs_xstory_bcp47[i]
        )
        for split in ["eval"]:
            translated_examples = []
            for example in tqdm(
                xstory_cloze[lang][split],
                total=len(xstory_cloze[lang][split]),
                desc=f"Translating {name} {lang}",
            ):
                translated_example = translate_example(example, tokenizer, model)
                translated_examples.append(translated_example)
            save_file(translated_examples, lang, split, name)


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
    model = load_model(model_name)
    translate_dataset(xstory_cloze, model, model_name)


if __name__ == "__main__":
    main()
