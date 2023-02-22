from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np

langs_xstory = ["en", "ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]


def load_model(model_name):
    """Load a model and tokenizer.

    Args:
        model_name (string): model name

    Returns:
        tokenizer, model: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.cuda()
    return tokenizer, model


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


def compute_perplexity(model, tokenizer, dataset):
    encodings = tokenizer("\n\n".join(dataset["text_right_ending"]), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len
            # compute bits per character: bpc is just log2(likekihood) / number-of-tokens
            # bpc = - np.log2(outputs.loss) / trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

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
    tokenizer, model = load_model(model_name)
    dataset_name = args.dataset_name
    xstory_cloze = get_dataset(dataset_name)
    perplexity = {}
    for lang in langs_xstory:
        print(f"Computing perplexity for {model} {lang}...")
        perplexity[lang] = compute_perplexity(model, tokenizer, xstory_cloze[lang]["eval"])
        print(f"Perplexity for {model} {lang}: {perplexity[lang]}")
    # save perplexity
    with open(f"../results/{name}_perplexity.txt", "w") as f:
        for lang in perplexity:
            f.write(f"{lang} {perplexity[lang]}")

    