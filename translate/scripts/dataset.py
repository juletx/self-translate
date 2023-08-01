from typing import List, Tuple
from torch.utils.data import IterableDataset


def count_lines(input_list: List[str]) -> int:
    """
    Counts the number of lines in a list of strings.

    Args:
        input_list (List[str]): List of strings.

    Returns:
        int: Number of lines in the list.
    """
    return len(input_list)


class DatasetReader(IterableDataset):
    def __init__(self, sentences: List[str], tokenizer, max_length: int = 128):
        """
        Initializes the DatasetReader class.

        Args:
            sentences (List[str]): List of sentences.
            tokenizer: Tokenizer object.
            max_length (int, optional): Maximum length of the tokenized sentence. Defaults to 128.
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.current_line = 0
        self.total_lines = count_lines(sentences)
        print(f"{self.total_lines} lines in list")

    def preprocess(self, text: str) -> dict:
        """
        Preprocesses a sentence by tokenizing it.

        Args:
            text (str): Input sentence.

        Returns:
            dict: Tokenized sentence.
        """
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
        mapped_itr = map(self.preprocess, self.sentences)
        return mapped_itr

    def __len__(self) -> int:
        return self.total_lines


class ParallelTextReader(IterableDataset):
    def __init__(self, predictions: List[str], references: List[str]):
        """
        Initializes the ParallelTextReader class.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[str]): List of reference sentences.
        """
        self.predictions = predictions
        self.references = references
        predictions_lines = count_lines(predictions)
        references_lines = count_lines(references)
        assert predictions_lines == references_lines, (
            f"Lines in predictions and references do not match "
            f"{predictions_lines} vs {references_lines}"
        )
        self.num_sentences = references_lines
        self.current_line = 0

    def preprocess(self, pred: str, gold: str) -> Tuple[str, List[str]]:
        """
        Preprocesses a predicted and a reference sentence by stripping them.

        Args:
            pred (str): Predicted sentence.
            gold (str): Reference sentence.

        Returns:
            Tuple[str, List[str]]: Tuple containing the predicted sentence and a list with the reference sentence.
        """
        self.current_line += 1
        pred = pred.rstrip().strip()
        gold = gold.rstrip().strip()
        if len(pred) == 0:
            print(f"Warning: Pred empty sentence at line {self.current_line}")
        if len(gold) == 0:
            print(f"Warning: Gold empty sentence at line {self.current_line}")
        return pred, [gold]

    def __iter__(self):
        mapped_itr = map(self.preprocess, self.predictions, self.references)
        return mapped_itr

    def __len__(self) -> int:
        return self.num_sentences
