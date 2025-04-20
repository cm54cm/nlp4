import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# -------------------- 数据加载 --------------------
def load_corpus(corpus_dir):
    """加载金庸小说语料库"""
    novels = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(corpus_dir, file_name), 'r', encoding='gb18030') as f:
                text = f.read().replace('\n', '').replace(' ', '')
            if text:
                novels.append(text)
    if novels:
        return novels[0]
    else:
        raise ValueError("没有找到有效的文本数据")

# -------------------- 数据集定义 --------------------
class TextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size):
        self.examples = []
        input_ids = tokenizer.encode(text)
        for i in range(0, len(input_ids) - block_size + 1, block_size):
            example = input_ids[i:i + block_size]
            if len(example) == block_size:
                self.examples.append(example)
            else:
                break
        print(f"Number of examples in dataset: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.examples[idx], dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": torch.ones_like(input_tensor),  # 全1表示没有padding
            "labels": input_tensor.clone()
        }


# -------------------- 配置参数 --------------------
model_name = "./local_gpt2_model"
corpus_dir = "your_corpus_directory"  # 替换为你的中文txt文件所在目录
block_size = 512
num_train_epochs = 3
per_device_train_batch_size = 4
learning_rate = 3e-4
output_dir = "fine-tuned-gpt2o"

# -------------------- 加载模型和分词器 --------------------
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -------------------- 配置LoRA --------------------
# 获取GPT - 2模型的层数
num_layers = model.config.n_layer
target_modules = []
for layer in range(num_layers):
    target_modules.extend([f"h.{layer}.attn.c_attn", f"h.{layer}.attn.c_proj"])

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- 加载数据集 --------------------
text = load_corpus(corpus_dir)
dataset = TextDataset(tokenizer, text, block_size)

# 创建数据加载器并测试
dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True)
for batch in dataloader:
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    break


# -------------------- 训练配置 --------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# -------------------- 开始训练 --------------------
trainer.train()

# -------------------- 保存微调后的模型 --------------------
model.save_pretrained(output_dir)

