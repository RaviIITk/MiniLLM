import re
import json
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab =vocab #total number of words where {int : word}
        self.rev_vocab = {v:k for k,v in self.vocab.items()}

    def encode(self,text):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text) #split the text into sentances\
        words = [item.strip() for item in words if item.strip()] # remove the spaces 
        words = [
            item for item in words                        # if any word is not present in vocab then putting "<|unk|>" 
        ]  

        return [self.vocab[s] for s in words]

    def decode(self, ids):
        text = " ".join([self.rev_vocab[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text               
        
if __name__=="__main__":
    print("fine")


        