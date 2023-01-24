# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XStoryCloze datasets."""


import csv
import os

import datasets


_DESCRIPTION = """
XStoryCloze consists of the professionally translated version of the [English StoryCloze dataset](https://cs.rochester.edu/nlp/rocstories/) (Spring 2016 version) to 10 non-English languages. This dataset is released by Meta AI.
"""

_CITATION = """\
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_LANG = ["en", "ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]
_URL_FORMAT = "spring2016.val.{lang}.tsv.split_20_80_{split}.tsv"


class XStoryCloze(datasets.GeneratorBasedBuilder):
    """XStoryCloze."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=lang,
                               description="XStoryCloze Test Spring 2016 {lang} set")
        for lang in _LANG
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "story_id": datasets.Value("string"),
                    "input_sentence_1": datasets.Value("string"),
                    "input_sentence_2": datasets.Value("string"),
                    "input_sentence_3": datasets.Value("string"),
                    "input_sentence_4": datasets.Value("string"),
                    "sentence_quiz1": datasets.Value("string"),
                    "sentence_quiz2": datasets.Value("string"),
                    "answer_right_ending": datasets.Value("int32"),
                }
            ),
            homepage="https://cs.rochester.edu/nlp/rocstories/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        name = self.config.name

        filepaths = dl_manager.download_and_extract({
            "train": _URL_FORMAT.format(lang=name, split="train"),
            "eval": _URL_FORMAT.format(lang=name, split="eval"),
        })

        return [
            datasets.SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": path},
            ) for split, path in filepaths.items()
        ]

    def _generate_examples(self, filepath):
        """Generate XStoryCloze examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                if row and len(row) == 8:
                    yield id_, {
                        "story_id": row[0],
                        "input_sentence_1": row[1],
                        "input_sentence_2": row[2],
                        "input_sentence_3": row[3],
                        "input_sentence_4": row[4],
                        "sentence_quiz1": row[5],
                        "sentence_quiz2": row[6],
                        "answer_right_ending": int(row[7]),
                    }
