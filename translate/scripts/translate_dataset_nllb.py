from collections import defaultdict
from datasets import load_dataset, DatasetDict
import translate
import os
import argparse
import pandas as pd
from dataset_configs import dataset_configs


def get_dataset(dataset_args):
    dataset = DatasetDict()
    for config in dataset_args["dataset_configs"]:
        dataset[config] = load_dataset(
            dataset_args["dataset"], config, split=dataset_args["dataset_split"]
        )
    return dataset


def get_texts(dataset, dataset_args):
    texts = defaultdict(dict)
    for config in dataset_args["dataset_configs"]:
        for field in dataset_args["dataset_fields"]:
            texts[config][field] = dataset[config][field]
    return texts


def translate_texts(dataset, texts, translate_args, dataset_args):
    translations = {}
    for config in dataset_args["dataset_configs"]:
        translations[config] = dataset[config].to_dict()
        translate_args["source_lang"] = dataset_args["lang_codes"][config]
        print(f"Translating from {config}")
        for field in dataset_args["dataset_fields"]:
            translations[config][field] = translate.main(
                sentences_list=texts[config][field],
                return_output=True,
                **translate_args,
            )
        save_file(translations[config], config, translate_args, dataset_args)


def save_file(translations, config, translate_args, dataset_args):
    name = translate_args["model_name"].split("/")[-1]
    dirname = f"{dataset_args['file_path']}/{name}"
    # create directory if it does not exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    translated_df = pd.DataFrame(translations)
    filename = f"{dirname}/{dataset_args['filename'].format(config=config)}"
    if filename.endswith(".tsv"):
        translated_df.to_csv(filename, sep="\t", index=False)
    elif filename.endswith(".jsonl"):
        translated_df.to_json(filename, orient="records", lines=True)
    else:
        raise ValueError("Unknown file format")


def main(translate_args, dataset_args):
    dataset = get_dataset(dataset_args)
    texts = get_texts(dataset, dataset_args)
    translate_texts(dataset, texts, translate_args, dataset_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the translation of a dataset or dict"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to translate, dump file path os huggingface identifier is supported",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language id. See: supported_languages.md",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=128,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
        "If you use multiple devices, we will divide this number by the number of devices.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_1.2B",
        help="Path to the model to use. See: https://huggingface.co/models",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory from which to load the model, or None to not cache",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokens in the source sentence and generated sentence. "
        "Increase this value to translate longer sentences, at the cost of increasing memory usage.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search, m2m10 author recommends 5, but it might use too much memory",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of possible translation to return for each sentence (num_return_sequences<=num_beams).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["bf16", "fp16", "32"],
        help="Precision of the model. bf16, fp16 or 32.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of beam search.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling, value used only if do_sample is True.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Keep special tokens in the decoded text.",
    )

    args = parser.parse_args()

    translate_args = dict(
        target_lang=args.target_lang,
        starting_batch_size=args.starting_batch_size,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        precision=args.precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        keep_special_tokens=args.keep_special_tokens,
    )

    dataset_args = dataset_configs[args.dataset]

    main(translate_args, dataset_args)
