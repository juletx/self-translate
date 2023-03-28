"""Calculate tranlation metrics for a given dataset"""
import evaluate
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

langs_xstory = ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]

chrf = evaluate.load("chrf")
sacrebleu = evaluate.load("sacrebleu")


def calculate_metrics(predictions, references):
    """Calculate metrics for a given dataset
    Args:
        predictions (list): list of predictions
        references (list): list of references
    Returns:
        chrf_results (dict): chrf results
        bleu_results (dict): bleu results
    """
    chrf_results = chrf.compute(
        predictions=predictions, references=references, word_order=2
    )
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    return chrf_results, bleu_results


def get_references():
    """Get references for a given dataset
    Returns:
        references (list): list of references
    """
    filepath = "../datasets/xstory_cloze/"
    filename = f"{filepath}/spring2016.val.en.tsv.split_20_80_eval.tsv"
    xstory_cloze_en = pd.read_csv(filename, sep="\t", na_filter=False)
    references = []
    for _, example in xstory_cloze_en.iterrows():
        references.extend(
            [
                [example["InputSentence1"]],
                [example["InputSentence2"]],
                [example["InputSentence3"]],
                [example["InputSentence4"]],
                [example["RandomFifthSentenceQuiz1"]],
                [example["RandomFifthSentenceQuiz2"]],
            ]
        )
    return references


def get_predictions(model, lang, folder):
    """Get predictions for a given dataset
    Args:
        model (str): model name
        lang (str): language
        folder (str): folder name
    Returns:
        predictions (list): list of predictions
    """
    filepath = f"../datasets/{folder}/{model}"
    filename = f"{filepath}/spring2016.val.{lang}.tsv.split_20_80_eval.tsv"
    xstory_cloze_lang = pd.read_csv(filename, sep="\t", na_filter=False)
    predictions = []
    for _, example in xstory_cloze_lang.iterrows():
        if model.startswith("xglm"):
            predictions.extend(
                [
                    example["input_sentence_1"].split(".")[0],
                    example["input_sentence_2"].split(".")[0],
                    example["input_sentence_3"].split(".")[0],
                    example["input_sentence_4"].split(".")[0],
                    example["sentence_quiz1"].split(".")[0],
                    example["sentence_quiz2"].split(".")[0],
                ]
            )
        else:
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
    """Main function"""
    references = get_references()
    model_names = {}
    model_names["xstory_cloze_mt"] = [
        "nllb-200-distilled-600M",
        "nllb-200-distilled-1.3B",
        "nllb-200-1.3B",
        "nllb-200-3.3B",
        "xglm-564M",
        "xglm-1.7B",
        "xglm-2.9B",
        "xglm-4.5B",
        "xglm-7.5B",
        "bloom-560m",
        "bloom-1b1",
        "bloom-1b7",
        "bloom-3b",
        "bloom-7b1",
        "bloomz-560m",
        "bloomz-1b1",
        "bloomz-1b7",
        "bloomz-3b",
        "bloomz-7b1",
        "bloomz-7b1-mt",
        "bloomz-7b1-p3",
    ]
    model_names["xstory_cloze_mt_few_shot"] = [
        "opt-125m",
        "opt-350m",
        "opt-1.3b",
        #"opt-2.7b",
        #"opt-6.7b",
        #"opt-13b",
        "xglm-564M",
        "xglm-1.7B",
        "xglm-2.9B",
        "xglm-4.5B",
        "xglm-7.5B",
        "bloom-560m",
        "bloom-1b1",
        "bloom-1b7",
        "bloom-3b",
        "bloom-7b1",
        "bloomz-560m",
        "bloomz-1b1",
        "bloomz-1b7",
        "bloomz-3b",
        "bloomz-7b1",
        "bloomz-7b1-mt",
        "bloomz-7b1-p3",
    ]
    for folder in ["xstory_cloze_mt_few_shot"]:
        results = {
            "model": [],
            "lang": [],
            "chrf": [],
            "bleu": [],
        }
        for model in model_names[folder]:
            avg_chrf = 0
            avg_bleu = 0
            for lang in langs_xstory:
                print("Evaluating", folder, model, lang)
                predictions = get_predictions(model, lang, folder)
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
        results_df.to_csv(f"translation_metrics_{folder}.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
