"""Evaluate a model on the xStoryCloze dataset.
"""
from argparse import ArgumentParser
import math
import torch
from tqdm import tqdm
import pandas as pd
from fairseq.models.transformer_lm import TransformerLanguageModel
from datasets import load_dataset


langs_xstory = ["en", "ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]


def load_model(model_name):
    """Load a model and tokenizer.

    Args:
        model_name (string): model name

    Returns:
        model: model
    """
    if model_name.endswith("bpe_model"):
        model = TransformerLanguageModel.from_pretrained(
            model_name,
            checkpoint_file="checkpoint.pt",
            bpe="fastbpe",
            bpe_codes=f"{model_name}/codes",
            dictionary=f"{model_name}/dict.txt",
            truncate_source=True,
            truncate_target=True,
        )
    else:
        model = TransformerLanguageModel.from_pretrained(
            model_name,
            checkpoint_file="checkpoint.pt",
            bpe="sentencepiece",
            sentencepiece_model=f"{model_name}/spm.model",
            sentencepiece_vocab=f"{model_name}/spm.vocab",
            dictionary=f"{model_name}/dict.txt",
            truncate_source=True,
            truncate_target=True,
        )
    model.eval()
    model.cuda()
    return model


def get_dataset(dataset_name):
    """Load the xStoryCloze dataset.

    Args:
        dataset_name (string): dataset name

    Returns:
        xstory_cloze: dataset
    """
    xstory_cloze = {}
    for lang in langs_xstory:
        xstory_cloze[lang] = load_dataset(dataset_name, lang)
    return xstory_cloze


def get_logprobs(prompt, model):
    """Get log probabilities of a prompt.

    Args:
        prompt (string): prompt
        model (model): model

    Returns:
        logprobs: log probabilities
    """
    logprobs = model.score(prompt, replace_newlines_with_eos=True)["positional_scores"]
    return logprobs


def xstory_cloze_eval(example, model):
    """Evaluate a model on a single example.

    Args:
        example (dict): example
        model (model): model

    Returns:
        pred, lprob1, lprob2, ppl1, ppl2: prediction, log probabilities and perplexities
    """
    input_sentences = (
        example["input_sentence_1"]
        + " "
        + example["input_sentence_2"]
        + " "
        + example["input_sentence_3"]
        + " "
        + example["input_sentence_4"]
    )
    prompt1 = input_sentences + " " + example["sentence_quiz1"]
    prompt2 = input_sentences + " " + example["sentence_quiz2"]
    lprob1 = get_logprobs(prompt1, model).mean()
    lprob2 = get_logprobs(prompt2, model).mean()
    ppl1 = torch.exp(-lprob1).item()
    ppl2 = torch.exp(-lprob2).item()
    lprob1i = lprob1.item()
    lprob2i = lprob2.item()
    pred = 1 if lprob1i > lprob2i else 2
    del lprob1, lprob2
    torch.cuda.empty_cache()
    return pred, lprob1i, lprob2i, ppl1, ppl2


def compute_results(xstory_cloze, model, model_name, dataset_name):
    """Evaluate a model on the xStoryCloze dataset and save results.

    Args:
        xstory_cloze (dataset): dataset
        model (model): model
        model_name (string): model name
        dataset_name (string): dataset name
    """
    size = len(xstory_cloze[langs_xstory[0]]["eval"])
    results_xstory = {
        "idx": list(range(size)),
        "label": xstory_cloze[langs_xstory[0]]["eval"]["answer_right_ending"],
    }
    for lang in langs_xstory:
        predictions, lprobs1, lprobs2, ppls1, ppls2 = [], [], [], [], []
        for _, example in tqdm(
            enumerate(xstory_cloze[lang]["eval"]),
            total=size,
            desc=f"Evaluating {model_name} on {lang}",
        ):
            pred, lprob1, lprob2, ppl1, ppl2 = xstory_cloze_eval(example, model)
            predictions.append(pred)
            lprobs1.append(round(lprob1, 2))
            lprobs2.append(round(lprob2, 2))
            ppls1.append(round(ppl1, 2))
            ppls2.append(round(ppl2, 2))
        results_xstory[lang] = predictions
        results_xstory[lang + "_lprob1"] = lprobs1
        results_xstory[lang + "_lprob2"] = lprobs2
        results_xstory[lang + "_ppl1"] = ppls1
        results_xstory[lang + "_ppl2"] = ppls2

    dname = dataset_name.split("/")[-1]
    pd.DataFrame(results_xstory).to_csv(
        f"../results/{dname}_{model_name}.tsv", sep="\t", index=False
    )


def get_accuracy(results_xstory_df):
    """Compute accuracy.

    Args:
        results_xstory_df (dataframe): dataframe

    Returns:
        accuracy (dict): accuracy
    """
    accuracy = {}
    for lang in langs_xstory:
        compare = results_xstory_df["label"] == results_xstory_df[lang]
        acc = list(compare).count(True) / len(list(compare)) * 100
        accuracy[lang] = round(acc, 1)
    accuracy["avg"] = round(sum(accuracy.values()) / len(accuracy), 1)
    return accuracy


def get_perplexity(results_xstory_df):
    """Compute perplexity.

    Args:
        results_xstory_df (dataframe): dataframe

    Returns:
        perplexity_correct (dict): perplexity for correct answers
        perplexity_incorrect (dict): perplexity for incorrect answers
    """
    perplexity_correct = {}
    perplexity_incorrect = {}

    for lang in langs_xstory:
        correct = []
        incorrect = []
        for _, row in results_xstory_df.iterrows():
            if row["label"] == 1:
                correct.append(-row[lang + "_lprob1"])
                incorrect.append(-row[lang + "_lprob2"])
            else:
                correct.append(-row[lang + "_lprob2"])
                incorrect.append(-row[lang + "_lprob1"])
        perplexity_correct[lang] = round(math.exp(sum(correct) / len(correct)), 2)
        perplexity_incorrect[lang] = round(math.exp(sum(incorrect) / len(incorrect)), 2)
    perplexity_correct["avg"] = round(
        sum(perplexity_correct.values()) / len(perplexity_correct), 2
    )
    perplexity_incorrect["avg"] = round(
        sum(perplexity_incorrect.values()) / len(perplexity_incorrect), 2
    )
    return perplexity_correct, perplexity_incorrect


def compute_metrics(model_name, dataset_name):
    """Compute metrics and save them.
    Args:
        model_name (string): model name
        dataset_name (string): dataset name
    """
    dname = dataset_name.split("/")[-1]
    results_xstory_df = pd.read_csv(
        f"../results/{dname}_{model_name}.tsv", delimiter="\t"
    )
    accuracy = get_accuracy(results_xstory_df)
    perplexity_correct, perplexity_incorrect = get_perplexity(results_xstory_df)
    metrics_df = pd.DataFrame(
        {
            "acc": accuracy,
            "ppl_cor": perplexity_correct,
            "ppl_inc": perplexity_incorrect,
        }
    )

    metrics_df.to_csv(
        f"../results/{dname}_{model_name}_metrics.tsv",
        sep="\t",
        index_label="lang",
    )


def main():
    """Main function."""
    parser = ArgumentParser("Evaluate a model on the xStoryCloze dataset")
    parser.add_argument(
        "model_name",
        type=str,
        help="Huggingface model name or path to a local model",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to use",
    )
    args = parser.parse_args()

    model_name = args.model_name
    name = model_name.split("/")[-1]
    model = load_model(model_name)
    dataset_name = args.dataset_name
    xstory_cloze = get_dataset(dataset_name)
    compute_results(xstory_cloze, model, name, dataset_name)
    compute_metrics(name, dataset_name)


if __name__ == "__main__":
    main()
