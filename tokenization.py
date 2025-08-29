import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()] #preprocessed contains the tokenized text
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)} #vocabulary created from the text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed] #replace unknow words by <unk> tokens
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) #replaces spaces before the specified punctuations
        return text
    
    tokenizer = tiktoken.get_encoding("gpt2") #instantiate the BPE tokenizer from tiktoken, similar use to SimpleTokenizerV2
                                            #BPE tokenizers break down unknown words into subwords and individual chars that it can parse ay word and doesn't need to replace with <unk>
    enc_text = tokenizer.encode(raw_text)
    enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        #print (tokenizer.decode(context), "---->", tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = [] 

        token_ids = tokenizer.encode(txt) # tokenizes the entire text

        for i in range(0, len(token_ids) - max_length, stride):  # uses a sliding window to chunck the book into overlapping sequences of max_length
            input_chunck = token_ids[i:i + max_length]
            target_chunck = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunck))
            self.target_ids.append(torch.tensor(target_chunck))

    def __len__(self):  # returns the total number of rows in the dataset
        return len(self.input_ids)
    
    def __getitem__(self, idx):  # returns a single row from the dataset
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last =drop_last,  # drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
        num_workers=num_workers  # the number of CPU processes to use for preprocessing
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Inputs:\n", inputs)
#print("\nTargets:\n", targets)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) 
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride= max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs) #embed the token ids into 256-dimensional vectors
#print(token_embeddings.shape) # will print size[8, 4, 256]

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # torch.arange() is a placeholder vector which contains a sequence of numbers 0, 1 ... up to the maximum input length - 1.
input_embeddings = token_embeddings + pos_embeddings  # the input_embeddings we created are the embedded input examples that can now be processed by the main LLM modules.
