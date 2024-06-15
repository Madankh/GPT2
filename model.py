import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

#----------------------

torch.manual_seed(1332)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt' , 'r' , encoding='utf-8') as f:
    text = f.read()

# Create a mapping from charaters to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda s : ''.join([itos[c] for c in s])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f'when input is {context} the target: {target}')


torch.manual_seed(1337)
batch_size = 4 # how many independent seq will we process in parallel?
block_size = 8  # what is the maximum context length for predictions ?

def get_batch(split):
  # generate a small batch of data of inputs x and target y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))

  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  x, y = x.to(device) , y.to(device)

  return x , y

xb , yb = get_batch('train')
print("Inputs:")
print(xb.shape)
print(xb)
print(yb, "yb")
print(yb.shape)

torch.manual_seed(1337)

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size , (batch_size,))

  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  x,y = x.to(device), y.to(device)
  return x,y


class BigramsLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a loopup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    # idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits , loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx)
      # print(logits.shape)
      # focus only on the last time step
      logits = logits[:,-1,:] # Becomes (B , C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B , C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramsLanguageModel(vocab_size)

n = model.to(device)
# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  xb , yb = get_batch