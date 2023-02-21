"""Train a tokenizer."""
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset


def get_dataset(dataset_name):
    """Load a dataset.
    Args:
        dataset_name (string): dataset name
    Returns:
        dataset: dataset
    """
    if dataset_name == "HiTZ/euscrawl":
        dataset = load_dataset(dataset_name)
    elif dataset_name == "mc4":
        dataset = load_dataset(dataset_name, "eu")
    else:
        dataset = load_dataset(dataset_name, lang="eu")
    if "validation" not in dataset.keys():
        dataset = split_dataset(dataset)
    return dataset


def split_dataset(dataset):
    """Split a dataset.
    Args:
        dataset (dict): dataset
    Returns:
        dataset_split: split dataset
    """
    # euscrawl and cc100 by default only contain the 'train' split, so create a test split
    dataset_split = dataset["train"].train_test_split(
        test_size=0.001, seed=42, shuffle=True
    )
    # rename the test split to validation
    dataset_split["validation"] = dataset_split.pop("test")
    return dataset_split


def get_training_corpus(raw_datasets, dataset_name):
    """Get training corpus.
    Args:
        raw_datasets (dict): raw datasets
    Returns:
        training_corpus: training corpus
    """
    field = "plain_text" if dataset_name == "HiTZ/euscrawl" else "text"
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples[field]


def train_tokenizer(training_corpus, model_name, tokenizer_name):
    """Train a tokenizer.
    Args:
        dataset_name (string): dataset name
        model_name (string): model name
        tokenizer_name (string): tokenizer name
    """

    # get old tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # train tokenizer
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=50304)

    # save tokenizer
    tokenizer.save_pretrained(f"out/{tokenizer_name}")

    # push tokenizer to hub
    tokenizer.push_to_hub(f"HiTZ/{tokenizer_name}")


def main():
    """Main function."""
    parser = ArgumentParser("Train a tokenizer")
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
    # load dataset
    print("Loading dataset...")
    dataset = get_dataset(args.dataset_name)
    # get training corpus
    print("Getting training corpus...")
    training_corpus = get_training_corpus(dataset, args.dataset_name)
    # train tokenizer
    print("Training tokenizer...")
    if args.dataset_name == "HiTZ/euscrawl":
        args.dataset_name = "euscrawl"
    tokenizer_name = f"{args.model_name}-eus-{args.dataset_name}"
    train_tokenizer(training_corpus, args.model_name, tokenizer_name)


if __name__ == "__main__":
    main()
