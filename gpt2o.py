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
num_train_epochs = 300
per_device_train_batch_size = 16
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

# -------------------- 文本生成函数 --------------------
def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    generated = input_ids

    for _ in range(max_length):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]

        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# -------------------- 测试生成 --------------------
prompt = "忽听得窗外有人格格轻笑，说道：“无忌哥哥，你可也曾答允了我做一件事啊。”正是周芷若的声音。张无忌凝神写信，竟不知她何时来到窗外。窗子缓缓推开，周芷若一张俏脸似笑非笑的现在烛光之下。张无忌惊道：“你……你又要叫我作甚么了？”周芷若微笑道：“这时候我还想不到。哪一日你要和赵家妹子拜堂成亲，只怕我便想到了。”张无忌回头向赵敏瞧了一眼，又回头向周芷若瞧了一眼，霎时之间百感交集，也不知是喜是忧，手一颤，一枝笔掉在桌上,"
output = generate_text(prompt, model, tokenizer, max_length=100)

print("=== 输入提示 ===")
print(prompt)
print("\n=== 生成内容 ===")
print(output)

# -------------------- 保存生成内容到 txt 文件 --------------------
output_file = "generated_text_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== 输入提示 ===\n")
    f.write(prompt + "\n\n")
    f.write("=== 生成内容 ===\n")
    f.write(output + "\n")

print(f"\n生成内容已保存到文件: {output_file}")

# -------------------- 测试生成 --------------------
prompt = "忽听得窗外有人格格轻笑，说道：“无忌哥哥，你可也曾答允了我做一件事啊，"
output = generate_text(prompt, model, tokenizer, max_length=100)

print("=== 输入提示 ===")
print(prompt)
print("\n=== 生成内容 ===")
print(output)

# -------------------- 保存生成内容到 txt 文件 --------------------
output_file = "generated_text_output1.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== 输入提示 ===\n")
    f.write(prompt + "\n\n")
    f.write("=== 生成内容 ===\n")
    f.write(output + "\n")

print(f"\n生成内容已保存到文件: {output_file}")