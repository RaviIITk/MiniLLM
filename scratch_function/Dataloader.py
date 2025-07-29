import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class MiniLLMTokenizer:
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []
    
        token_ids = tokenizer.encode(text)
        context_size= min(len(token_ids)-2,context_size )
        
        
        for i in range(1, len(token_ids)- context_size, stride):
            input_chunk = token_ids[i:i + context_size]
            target_chunk = token_ids[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def data_loader(text, batch_size =8, context_size=256, stride=128, shuffle = True, drop_last = True, n_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')
    
    dataset = MiniLLMTokenizer(text, tokenizer,context_size,stride=stride)
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=n_workers
    )

    return dataloader


if __name__=='__main__':
    text = 'How are you?, how you doing?, its been so long i havnt seen you'
    print(print(next(iter(data_loader(text, batch_size=1,context_size=8, stride=1)))))