import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import os



class TextForMultiHeadCausalLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        """
        texts: list of str - 入力テキスト
        tokenizer: Hugging Face トークナイザー
        max_length: 最大トークン長
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # トークナイズ & パディング
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ラベルは input_ids のコピー
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloader(tokenizer, batch_size=2, max_length=32, n=5):
    """
    デモ用: 実際には load_dataset に適切なパス等を指定してください。
    """
    def apply_chat_format(example):
        return {
            "text": tokenizer.apply_chat_template([{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["reference_answer"]}], tokenize=False)
        }
    dataset = load_dataset("facebook/natural_reasoning", split="train").map(apply_chat_format)
    dataset = dataset.train_test_split(test_size=100)
    train_dataset = dataset["train"]
    test_dataset  = dataset["test"]

    train_texts = train_dataset["text"]
    test_texts  = test_dataset["text"]

    train_dataset = TextForMultiHeadCausalLMDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = TextForMultiHeadCausalLMDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(
    model,
    train_loader,
    eval_loader,
    accelerator,
    optimizer,
    save_path,
    log_steps,
    eval_and_save_steps,
    num_epochs=1,
):
    """
    モデルの学習を行う関数
    """
    os.makedirs(save_path, exist_ok=True)

    # optimizerはすでにaccelerator.prepareで準備済み
    device = accelerator.device

    wandb.init(project="farsight", name="multi-head-training")

    global_step = 0
    progress_bar = trange(num_epochs, desc="Epoch")

    for epoch in progress_bar:
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Train Iter")

        for step, batch in enumerate(epoch_iterator):
            # with accelerator.accumulate(model):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            accelerator.backward(loss)

            if (step + 1) % log_steps == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/loss_head1": outputs.loss_head1.item(),
                        "train/loss_head2": outputs.loss_head2.item(),
                    },
                    step=global_step,
                )

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % eval_and_save_steps == 0:
                evaluate(model, eval_loader, device, save_path, global_step)
                if accelerator and accelerator.is_main_process:
                    accelerator.save_state(f"{save_path}/checkpoint-{global_step}")
                elif not accelerator:
                    model.save_pretrained(f"{save_path}/checkpoint-{global_step}")

    wandb.finish()


def evaluate(model, eval_loader, device, save_path, global_step):
    model.eval()
    total_eval_loss = 0.0
    total_eval_lm_loss = 0.0
    total_eval_n_step_loss = 0.0

    # n-step予測の正答率などを測るならここで計算もできるが、生成タスクの場合は難しいので割愛
    # ここではデモとして「n-step 用の正解トークンと予測トークンが一致したか」を計算し、
    # accuracy を出すようにしてみる
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval Iter"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            n_step_labels = batch["n_step_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                n_step_labels=n_step_labels,
                return_dict=True,
            )

            loss = outputs.loss
            lm_loss = outputs.lm_loss
            n_step_loss = outputs.n_step_loss

            total_eval_loss += loss.item()
            total_eval_lm_loss += lm_loss.item()
            total_eval_n_step_loss += n_step_loss.item()

            # n_step_logits -> (batch, seq_len, vocab_size)
            # モデル出力から n-step ahead の予測分布を取り、argmax
            n_step_logits = outputs.n_step_logits  # カスタムモデル実装で用意
            preds = torch.argmax(n_step_logits, dim=-1)  # (batch, seq_len)

            # -100 は無視 (paddingなど)
            mask = n_step_labels != -100
            masked_preds = preds[mask]
            masked_labels = n_step_labels[mask]

            correct += (masked_preds == masked_labels).sum().item()
            total += masked_labels.numel()

    avg_eval_loss = total_eval_loss / len(eval_loader)
    avg_eval_lm_loss = total_eval_lm_loss / len(eval_loader)
    avg_eval_n_step_loss = total_eval_n_step_loss / len(eval_loader)

    acc = correct / total if total > 0 else 0.0

    print(
        f"[Eval] Global Step: {global_step}, "
        f"Loss: {avg_eval_loss:.4f}, "
        f"LM Loss: {avg_eval_lm_loss:.4f}, "
        f"N-step Loss: {avg_eval_n_step_loss:.4f}, "
        f"N-step Accuracy: {acc:.4f}"
    )

    wandb.log(
        {
            "Eval/Loss": avg_eval_loss,
            "Eval/LM Loss": avg_eval_lm_loss,
            "Eval/N-step Loss": avg_eval_n_step_loss,
            "Eval/N-step Accuracy": acc,
        },
        step=global_step,
    )

    # 任意: モデルをステップごとに保存
    model.save_pretrained(os.path.join(save_path, f"step_{global_step}"))

    return avg_eval_loss
