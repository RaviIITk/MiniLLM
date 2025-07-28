import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


class BPETokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_merges = []

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def build_vocab(self, corpus: List[str]):
        # Convert corpus to word frequencies
        word_freqs = Counter(corpus)
        self.vocab = {" ".join(list(word)) + " </w>": freq for word, freq in word_freqs.items()}
        
        # Extract initial tokens
        tokens = Counter()
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                tokens[(symbols[i], symbols[i+1])] += freq

        while len(self.bpe_merges) < self.vocab_size and tokens:
            # Find most frequent pair
            best = max(tokens, key=tokens.get)
            self.bpe_merges.append(best)
            self.vocab = self._merge_vocab(best)
            tokens = self._get_stats()

    def _get_stats(self) -> Dict[Tuple[str, str], int]:
        stats = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                stats[(symbols[i], symbols[i+1])] += freq
        return stats

    def _merge_vocab(self, pair: Tuple[str, str]) -> Dict[str, int]:
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        new_vocab = {}
        for word, freq in self.vocab.items():
            new_word = re.sub(rf'\b{pattern}\b', replacement, word)
            new_vocab[new_word] = freq
        return new_vocab

    def encode(self, word: str) -> List[str]:
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            merge_candidate = None
            for merge in self.bpe_merges:
                if merge in pairs:
                    merge_candidate = merge
                    break
            if not merge_candidate:
                break
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == merge_candidate:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def decode(self, tokens: List[str]) -> str:
        return ''.join([t.replace('</w>', '') for t in tokens])


# Example usage
if __name__ == "__main__":
    corpus = ["low", "lower", "newest", "widest"]
    tokenizer = BPETokenizer(vocab_size=10)
    tokenizer.build_vocab(corpus)

    print("BPE merges:", tokenizer.bpe_merges)

    encoded = tokenizer.encode("lowest")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)