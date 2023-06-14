import json
import os
from typing import List, Dict, DefaultDict, Optional
from collections import defaultdict

import torch
from accelerate import find_executable_batch_size
from comet import download_model, load_from_checkpoint
import evaluate
from dataset_configs import dataset_configs
from datasets import load_dataset, DatasetDict


_DATASETS = [
    # "xnli",
    # "xstory_cloze",
    # "mgsm",
    "xcopa",
    "pawsx",
]

_MODELS = [
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
    "llama-7B",
    "llama-13B",
    "llama-30B",
    # "llama-65B",
    "RedPajama-INCITE-Base-3B-v1",
    "RedPajama-INCITE-7B-Base",
    "open_llama_3b",
    "open_llama_7b",
]


def get_dataset(dataset_args: Dict[str, str]) -> DatasetDict:
    """
    Loads the dataset using the dataset_args.

    Args:
    - dataset_args (dict): A dictionary containing the dataset name, split, and configurations.

    Returns:
    - dataset (DatasetDict): A dictionary containing the dataset.
    """
    dataset = DatasetDict()
    if dataset_args["dataset"] == "xcopa":
        dataset["en"] = load_dataset(
            "super_glue", "copa", split=dataset_args["dataset_split"]
        )
    else:
        dataset["en"] = load_dataset(
            dataset_args["dataset"], "en", split=dataset_args["dataset_split"]
        )
    for config in dataset_args["dataset_configs"]:
        dataset[config] = load_dataset(
            dataset_args["dataset"], config, split=dataset_args["dataset_split"]
        )
    return dataset


def get_dataset_mt(dataset_args: Dict[str, str], model: str) -> DatasetDict:
    """
    Loads the machine translation dataset using the dataset_args and model.

    Args:
    - dataset_args (dict): A dictionary containing the dataset name, split, and configurations.
    - model (str): The name of the model.

    Returns:
    - dataset (DatasetDict): A dictionary containing the machine translation dataset.
    """
    dataset = DatasetDict()
    for config in dataset_args["dataset_configs"]:
        dataset[config] = load_dataset(dataset_args["dataset_mt"], model, split=config)
    return dataset


def get_texts(
    dataset: DatasetDict, dataset_args: Dict[str, str]
) -> DefaultDict[str, Dict[str, List[str]]]:
    """
    Extracts the texts from the dataset.

    Args:
    - dataset (DatasetDict): A dictionary containing the dataset.
    - dataset_args (dict): A dictionary containing the dataset name, split, and configurations.

    Returns:
    - texts (defaultdict): A dictionary containing the texts for each configuration and field.
    """
    texts = defaultdict(dict)
    for config in dataset:
        for field in dataset_args["dataset_fields"]:
            texts[config][field] = dataset[config][field]
    return texts


def load_comet(model_name: str = "Unbabel/wmt22-comet-da"):
    """
    Loads the COMET model from a checkpoint.

    Args:
    - model_name (str): The name of the COMET model.

    Returns:
    - model: The loaded COMET model.
    """
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model


@find_executable_batch_size(starting_batch_size=2048)
def compute_comet(
    batch_size: int,
    model: load_from_checkpoint,
    predictions: List[str],
    references: List[str],
    sources: List[str],
    gpus: Optional[int] = None,
    progress_bar: bool = False,
) -> Dict[str, float]:
    """
    Computes the COMET score for a batch of translations.

    Args:
    - batch_size (int): The batch size for the COMET model.
    - model (load_from_checkpoint): The loaded COMET model.
    - predictions (List[str]): A list of translated sentences.
    - references (List[str]): A list of reference sentences.
    - sources (List[str]): A list of source sentences.
    - gpus (int): The number of GPUs to use for computation.
    - progress_bar (bool): Whether to display a progress bar during computation.

    Returns:
    - model_output (Dict[str, float]): A dictionary containing the COMET score for each sentence.
    """
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else 0
    data = {"src": sources, "mt": predictions, "ref": references}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    model_output = model.predict(
        data, batch_size=batch_size, gpus=gpus, progress_bar=progress_bar
    )
    return model_output


def evaluate_translations(
    predictions: List[str], references: List[str], sources: List[str]
) -> Dict[str, float]:
    """
    Evaluates the translations using sacrebleu, chrf and comet metrics.

    Args:
    - predictions (List[str]): A list of predicted translations.
    - references (List[str]): A list of reference translations.
    - sources (List[str]): A list of source sentences.

    Returns:
    - result_dictionary (Dict[str, float]): A dictionary containing the evaluation results for each metric.
    """
    print("Loading sacrebleu...")
    sacrebleu = evaluate.load("sacrebleu")

    print("Loading chrf...")
    chrf = evaluate.load("chrf")

    print("Loading comet...")
    model = load_comet("Unbabel/wmt22-comet-da")

    result_dictionary = {}
    print(f"Computing sacrebleu")
    sacrebleu_results = sacrebleu.compute(
        predictions=predictions, references=references
    )
    result_dictionary["sacrebleu"] = round(sacrebleu_results["score"], 2)

    print(f"Computing chrf score")
    chrf_results = chrf.compute(
        predictions=predictions, references=references, word_order=2
    )
    result_dictionary["chrf++"] = round(chrf_results["score"], 2)

    print("Computing comet score")
    comet_results = compute_comet(
        model=model,
        predictions=predictions,
        references=references,
        sources=sources,
        progress_bar=True,
    )
    comet_results["mean_score"] = sum(comet_results["scores"]) / len(
        comet_results["scores"]
    )
    result_dictionary["comet"] = round(comet_results["mean_score"] * 100, 2)

    print(result_dictionary)

    return result_dictionary


def evaluate_texts(
    predictions: DefaultDict[str, Dict[str, List[str]]],
    references: DefaultDict[str, Dict[str, List[str]]],
    dataset_args: Dict[str, str],
    model_name: str,
) -> None:
    """
    Evaluates the translations for each configuration and field.

    Args:
    - predictions (defaultdict): A dictionary containing the predicted translations for each configuration and field.
    - references (defaultdict): A dictionary containing the reference translations for each configuration and field.
    - dataset_args (dict): A dictionary containing the dataset name, split, and configurations.
    - model_name (str): The name of the model.
    """
    evaluations = {}
    for config in predictions:
        evaluations[config] = {}
        print(f"Evaluating config {config}")
        for field in dataset_args["dataset_fields"]:
            print(f"Evaluating field {field}")
            evaluations[config][field] = evaluate_translations(
                predictions=predictions[config][field],
                references=references["en"][field],
                sources=references[config][field],
            )
    save_file(evaluations, dataset_args, model_name)


def save_file(
    evaluations: Dict[str, Dict[str, Dict[str, float]]],
    dataset_args: Dict[str, str],
    model_name: str,
) -> None:
    """
    Saves the evaluation results to a file.

    Args:
    - evaluations (dict): A dictionary containing the evaluation results for each configuration and field.
    - dataset_args (dict): A dictionary containing the dataset name, split, and configurations.
    - model_name (str): The name of the model.
    """
    dirname = f"metrics/{dataset_args['dataset'].split('/')[-1]}"
    # create directory if it does not exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = f"{dirname}/{model_name}.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(evaluations, file, indent=2)


def main() -> None:
    """
    Main function that evaluates the translations for each dataset and model.
    """
    for dataset_name in _DATASETS:
        dataset_args = dataset_configs[dataset_name]
        print("Evaluating dataset", dataset_name)
        dataset = get_dataset(dataset_args)
        references = get_texts(dataset, dataset_args)
        for model_name in _MODELS:
            if model_name == "bloom-560m" and dataset_name == "xnli":
                continue
            print("Evaluating model", model_name)
            dataset_mt = get_dataset_mt(dataset_args, model_name)
            predictions = get_texts(dataset_mt, dataset_args)
            evaluate_texts(predictions, references, dataset_args, model_name)


if __name__ == "__main__":
    main()
