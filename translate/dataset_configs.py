dataset_configs = {
    "xnli": {
        "dataset": "xnli",
        "dataset_configs": [
            "ar",
            "bg",
            "de",
            "el",
            "es",
            "fr",
            "hi",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        ],
        "dataset_split": "test",
        "dataset_fields": ["premise", "hypothesis"],
        "file_path": "../datasets/xnli_mt",
        "filename": "xnli.{config}.test.tsv",
        "lang_codes": {
            "ar": "arb_Arab",
            "bg": "bul_Cyrl",
            "de": "deu_Latn",
            "el": "ell_Grek",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "hi": "hin_Deva",
            "ru": "rus_Cyrl",
            "sw": "swh_Latn",
            "th": "tha_Thai",
            "tr": "tur_Latn",
            "ur": "urd_Arab",
            "vi": "vie_Latn",
            "zh": "zho_Hans",
        },
        "lang_names": {
            "ar": "Arabic",
            "bg": "Bulgarian",
            "de": "German",
            "el": "Greek",
            "es": "Spanish",
            "fr": "French",
            "hi": "Hindi",
            "ru": "Russian",
            "sw": "Swahili",
            "th": "Thai",
            "tr": "Turkish",
            "ur": "Urdu",
            "vi": "Vietnamese",
            "zh": "Chinese",
        },
    },
    "xstory_cloze": {
        "dataset": "juletxara/xstory_cloze",
        "dataset_configs": ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"],
        "dataset_split": "eval",
        "dataset_fields": [
            "input_sentence_1",
            "input_sentence_2",
            "input_sentence_3",
            "input_sentence_4",
            "sentence_quiz1",
            "sentence_quiz2",
        ],
        "file_path": "../datasets/xstory_cloze_mt",
        "filename": "spring2016.val.{config}.tsv.split_20_80_eval.tsv",
        "lang_codes": {
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "es": "spa_Latn",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "id": "ind_Latn",
            "te": "tel_Telu",
            "sw": "swh_Latn",
            "eu": "eus_Latn",
            "my": "mya_Mymr",
        },
        "lang_names": {
            "ru": "Russian",
            "zh": "Chinese",
            "es": "Spanish",
            "ar": "Arabic",
            "hi": "Hindi",
            "id": "Indonesian",
            "te": "Telugu",
            "sw": "Swahili",
            "eu": "Basque",
            "my": "Burmese",
        },
    },
    "mgsm": {
        "dataset": "juletxara/mgsm",
        "dataset_configs": [
            "es",
            "fr",
            "de",
            "ru",
            "zh",
            "ja",
            "th",
            "sw",
            "bn",
            "te",
        ],
        "dataset_split": "test",
        "dataset_fields": ["question"],
        "file_path": "../datasets/mgsm_mt",
        "filename": "mgsm_{config}.tsv",
        "lang_codes": {
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "th": "tha_Thai",
            "sw": "swh_Latn",
            "bn": "ben_Beng",
            "te": "tel_Telu",
        },
        "lang_names": {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "th": "Thai",
            "sw": "Swahili",
            "bn": "Bengali",
            "te": "Telugu",
        },
    },
    "xcopa": {
        "dataset": "xcopa",
        "dataset_configs": [
            "et",
            "ht",
            "it",
            "id",
            "qu",
            "sw",
            "zh",
            "ta",
            "th",
            "tr",
            "vi",
        ],
        "dataset_split": "test",
        "dataset_fields": ["premise", "choice1", "choice2"],
        "file_path": "../datasets/xcopa_mt",
        "filename": "test.{config}.jsonl",
        "lang_codes": {
            "et": "est_Latn",
            "ht": "hat_Latn",
            "it": "ita_Latn",
            "id": "ind_Latn",
            "qu": "quy_Latn",
            "sw": "swh_Latn",
            "zh": "zho_Hans",
            "ta": "tam_Taml",
            "th": "tha_Thai",
            "tr": "tur_Latn",
            "vi": "vie_Latn",
        },
        "lang_names": {
            "et": "Estonian",
            "ht": "Haitian",
            "it": "Italian",
            "id": "Indonesian",
            "qu": "Quechua",
            "sw": "Swahili",
            "zh": "Chinese",
            "ta": "Tamil",
            "th": "Thai",
            "tr": "Turkish",
            "vi": "Vietnamese",
        },
    },
    "pawsx": {
        "dataset": "paws-x",
        "dataset_configs": [
            "de",
            "es",
            "fr",
            "ja",
            "ko",
            "zh",
        ],
        "dataset_split": "test",
        "dataset_fields": ["sentence1", "sentence2"],
        "file_path": "../datasets/pawsx_mt",
        "filename": "{config}_test_2k.tsv",
        "lang_codes": {
            "de": "deu_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "ja": "jpn_Jpan",
            "ko": "jpn_Hang",
            "zh": "zho_Hans",
        },
        "lang_names": {
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
        },
    },
    "xwinograd": {
        "dataset": "juletxara/xwinograd",
        "dataset_configs": ["fr", "jp", "pt", "ru", "zh"],
        "dataset_split": "test",
        "dataset_fields": ["sentence1", "sentence2"],
        "file_path": "../datasets/xwinograd_mt",
        "filename": "{config}.jsonl",
        "lang_codes": {
            "fr": "fra_Latn",
            "jp": "jpn_Jpan",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
        },
        "lang_names": {
            "fr": "French",
            "jp": "Japanese",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
        },
    },
}