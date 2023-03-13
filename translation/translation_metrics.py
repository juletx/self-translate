"""Calculate tranlation metrics for a given dataset"""
import evaluate
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

langs_xstory = ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]

chrf = evaluate.load("chrf")
sacrebleu = evaluate.load("sacrebleu")


def calculate_metrics(predictions, references):
    chrf_results = chrf.compute(
        predictions=predictions, references=references, word_order=2
    )
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    return chrf_results, bleu_results


def get_references():
    xstory_cloze_en = load_dataset("juletxara/xstory_cloze", "en", split="eval")
    references = []
    for example in xstory_cloze_en:
        references.extend(
            [
                [example["input_sentence_1"]],
                [example["input_sentence_2"]],
                [example["input_sentence_3"]],
                [example["input_sentence_4"]],
                [example["sentence_quiz1"]],
                [example["sentence_quiz2"]],
            ]
        )
    return references


def get_predictions(model, lang):
    filepath = f"../datasets/xstory_cloze_mt/{model}"
    filename = f"{filepath}/spring2016.val.{lang}.tsv.split_20_80_eval.tsv"
    xstory_cloze_lang = pd.read_csv(filename, sep="\t")
    predictions = []
    for index, example in xstory_cloze_lang.iterrows():
        predictions.extend(
            [
                example["input_sentence_1"],
                example["input_sentence_2"],
                example["input_sentence_3"],
                example["input_sentence_4"],
                example["sentence_quiz1"],
                example["sentence_quiz2"],
            ]
        )
    return predictions


def main():
    references = get_references()
    model_names = [
        "nllb-200-distilled-600M",
        "nllb-200-distilled-1.3B",
        "nllb-200-1.3B",
        "nllb-200-3.3B",
        "bloomz-560m",
        "bloomz-1b1",
        "bloomz-1b7",
        "bloomz-3b",
        "bloomz-7b1",
        "bloomz-7b1-mt",
        #"bloomz-7b1-p3",
    ]
    results = {
        "model": [],
        "lang": [],
        "chrf": [],
        "bleu": [],
    }
    for model in tqdm(model_names):
        avg_chrf = 0
        avg_bleu = 0
        for lang in langs_xstory:
            predictions = get_predictions(model, lang)
            chrf_results, bleu_results = calculate_metrics(predictions, references)
            results["model"].append(model)
            results["lang"].append(lang)
            results["chrf"].append(round(chrf_results["score"], 2))
            results["bleu"].append(round(bleu_results["score"], 2))
            avg_chrf += chrf_results["score"]
            avg_bleu += bleu_results["score"]
        results["model"].append(model)
        results["lang"].append("avg")
        results["chrf"].append(round(avg_chrf / len(langs_xstory), 2))
        results["bleu"].append(round(avg_bleu / len(langs_xstory), 2))
    results_df = pd.DataFrame(results)
    results_df.to_csv("translation_metrics.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
