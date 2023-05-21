from torch.utils.data import IterableDataset


def count_lines(input_list: list) -> int:
    return len(input_list)


class DatasetReader(IterableDataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.current_line = 0
        self.total_lines = count_lines(sentences)
        print(f"{self.total_lines} lines in list")

    def preprocess(self, text: str):
        self.current_line += 1
        text = text.rstrip().strip()
        if len(text) == 0:
            print(f"Warning: empty sentence at line {self.current_line}")
        return self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

    def __iter__(self):
        # file_itr = open(self.filename, "r", encoding="utf8")
        mapped_itr = map(self.preprocess, self.sentences)
        return mapped_itr

    def __len__(self):
        return self.total_lines


class ParallelTextReader(IterableDataset):
    def __init__(self, pred_path: str, gold_path: str):
        self.pred_path = pred_path
        self.gold_path = gold_path
        pref_filename_lines = count_lines(pred_path)
        gold_path_lines = count_lines(gold_path)
        assert pref_filename_lines == gold_path_lines, (
            f"Lines in {pred_path} and {gold_path} do not match "
            f"{pref_filename_lines} vs {gold_path_lines}"
        )
        self.num_sentences = gold_path_lines
        self.current_line = 0

    def preprocess(self, pred: str, gold: str):
        self.current_line += 1
        pred = pred.rstrip().strip()
        gold = gold.rstrip().strip()
        if len(pred) == 0:
            print(f"Warning: Pred empty sentence at line {self.current_line}")
        if len(gold) == 0:
            print(f"Warning: Gold empty sentence at line {self.current_line}")
        return pred, [gold]

    def __iter__(self):
        pred_itr = open(self.pred_path, "r", encoding="utf8")
        gold_itr = open(self.gold_path, "r", encoding="utf8")
        mapped_itr = map(self.preprocess, pred_itr, gold_itr)
        return mapped_itr

    def __len__(self):
        return self.num_sentences
