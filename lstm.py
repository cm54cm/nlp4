import os
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# -------------------- 数据加载 --------------------
def load_corpus(corpus_dir):
    """加载金庸小说语料库"""
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='gb18030') as f:
                text = f.read().replace('\n', '').replace(' ', '')
            novels.append(text)
    return novels[0]

# -------------------- 数据集定义 --------------------
class TextDataset(Dataset):
    def __init__(self, tokens, seq_len=100):
        self.seq_len = seq_len
        self.tokens = tokens
        # 建立字表
        self.vocab = sorted(set(tokens))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        # 编码所有 tokens
        self.data = [self.stoi[ch] for ch in tokens]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
        return x, y

# -------------------- LSTM 模型定义 --------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)
        out, h = self.lstm(emb, h)
        logits = self.fc(out)
        return logits, h

# -------------------- 生成函数 --------------------
@torch.no_grad()
def generate_lstm(model, dataset, prompt, length=200, device='cpu'):
    model.to(device)
    model.eval()
    # 初始化 hidden
    h = None
    # 用 prompt 隐藏状态
    for ch in prompt:
        if ch not in dataset.stoi:
            continue
        idx = torch.tensor([[dataset.stoi[ch]]], device=device)
        _, h = model(idx, h)
    # 从 prompt 最后一个字符继续生成
    last = prompt[-1] if prompt and prompt[-1] in dataset.stoi else dataset.itos[0]
    idx = torch.tensor([[dataset.stoi[last]]], device=device)
    out = list(prompt)
    for _ in range(length):
        logits, h = model(idx, h)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        ch = dataset.itos[idx.item()]
        out.append(ch)
    return ''.join(out)

# -------------------- 训练函数 --------------------
def train_lstm(corpus_dir, seq_len=100, embed_size=256, hidden_size=512,
               epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    # 加载数据
    tokens = load_corpus(corpus_dir)
    dataset = TextDataset(tokens, seq_len)
    print(f"数据总长度: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型与优化器
    model = LSTMModel(len(dataset.vocab), embed_size, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0
        prog = tqdm(loader, desc=f"[LSTM Epoch {ep}/{epochs}]")
        for x, y in prog:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {ep} 完成，平均 Loss: {avg:.4f}")

    # 保存权重
    torch.save(model.state_dict(), 'lstm_weights.pth')
    print('LSTM 权重已保存至 lstm_weights.pth')
    return model, dataset

# -------------------- 数据加载 --------------------
def load_corpus1(corpus_dir):
    """加载金庸小说语料库"""
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='gb18030') as f:
                text = f.read().replace('\n', '').replace(' ', '')
            novels.append(text)
    return novels
    
# -------------------- BERT 定义（本地模型） --------------------
class TransformerTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        all_ids = []
        for txt in texts:
            enc = tokenizer(txt, add_special_tokens=False)
            all_ids.extend(enc['input_ids'])
        self.examples = [
            torch.tensor(all_ids[i:i+block_size], dtype=torch.long)
            for i in range(0, len(all_ids)-block_size, block_size)
        ]
    def __len__(self): 
        print(len(self.examples))
        return len(self.examples)
    def __getitem__(self, idx):
        return {'input_ids': self.examples[idx], 'labels': self.examples[idx]}

@torch.no_grad()
def generate_bert(model, tokenizer, prompt, length=200, device='cpu'):
    model.to(device)
    model.eval()
    seq = prompt
    for _ in range(length):
        tokens = tokenizer(seq, return_tensors='pt', add_special_tokens=False)
        input_ids = tokens.input_ids[0].tolist() + [tokenizer.mask_token_id]
        attn_mask = [1] * len(input_ids)
        inputs = {'input_ids': torch.tensor([input_ids], device=device), 'attention_mask': torch.tensor([attn_mask], device=device)}
        logits = model(**inputs).logits
        mask_logits = logits[0, -1]
        idx = torch.multinomial(torch.softmax(mask_logits, dim=-1), num_samples=1).item()
        token = tokenizer.convert_ids_to_tokens(idx).replace('##','')
        seq += token
    return seq

# -------------------- 主流程 --------------------
def main():
    corpus_dir = './jyxstxtqj_downcc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # 1. BERT 微调 + 生成（本地模型）
    # print('== BERT MaskedLM 微调（本地模型） ==')
    # local_bert_dir = './bert_local'
    # # 准备好以下文件于 bert_local/: config.json, pytorch_model.bin, vocab.txt, tokenizer_config.json, special_tokens_map.json
    # tokenizer = BertTokenizerFast.from_pretrained(local_bert_dir, local_files_only=True)
    # model = BertForMaskedLM.from_pretrained(local_bert_dir, local_files_only=True)
    # trainer = Trainer(
    #     model=model,
    #     args=TrainingArguments(
    #         output_dir='./bert_output', overwrite_output_dir=True,
    #         num_train_epochs=30, per_device_train_batch_size=64, learning_rate=5e-5,
    #         logging_steps=50, save_steps=500, save_total_limit=2
    #     ),
    #     train_dataset=TransformerTextDataset(load_corpus1(corpus_dir), tokenizer, block_size=128),
    #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    # )
    # trainer.train()
    # model.save_pretrained('./bert_output')
    # tokenizer.save_pretrained('./bert_output')
    # prompt_bert = '“无忌哥哥，你可也曾答允了我做一件事啊。”'
    # print(f'BERT 预设提示词: {prompt_bert}')
    # print('BERT 生成示例:\n', generate_bert(model, tokenizer, prompt_bert, length=200, device=device))

    # 训练 LSTM
    print('== 开始 LSTM 从头训练 ==')
    model, dataset = train_lstm(corpus_dir,
                                seq_len=100,
                                embed_size=256,
                                hidden_size=512,
                                epochs=100,
                                batch_size=512,
                                lr=1e-3,
                                device=device)
    # 生成示例
    prompt = '“无忌哥哥，你可也曾答允了我做一件事啊。”'
    print(f'预设提示词：{prompt}')
    sample = generate_lstm(model, dataset, prompt, length=200, device=device)
    print('LSTM 生成示例:\n', sample)

if __name__ == '__main__':
    main()