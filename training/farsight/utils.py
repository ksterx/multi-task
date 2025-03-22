import torch
import torch.optim as optim
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    dataset = load_dataset("facebook/natural_reasoning", split="train[:20]").map(apply_chat_format)
    dataset = dataset.train_test_split(test_size=10)
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
    save_path="./",
    log_steps=100,
    eval_and_save_steps=500,
    num_epochs=3,
    lr=1e-5,
):
    """
    next-token (LM) loss + n-step先予測 loss を合算して学習。
    """
    wandb.init(project="multi-task", name="gemma2_n_step_pred", entity="ksterx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    global_step = 0
    best_eval_loss = float("inf")

    for epoch in trange(num_epochs, desc="Epoch"):
        model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_n_step_loss = 0.0

        for batch in tqdm(train_loader, desc="Train Iter"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)            # next-token 用
            n_step_labels = batch["n_step_labels"].to(device)  # n-step先

            optimizer.zero_grad()

            # モデルの forward:
            #  カスタムモデルであれば、n_step_labels を引数にして2つのロスを返すようにする
            #  例:
            #  outputs = model(
            #      input_ids=input_ids,
            #      attention_mask=attention_mask,
            #      labels=labels,
            #      n_step_labels=n_step_labels,
            #      return_dict=True,
            #  )
            #
            # 以下は「lm_loss」「n_step_loss」「loss」を全部返す想定。
            # ここではデモなので、その処理を想定して書きます。

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                n_step_labels=n_step_labels,  # カスタム引数
                return_dict=True,
            )

            # トータルロス + 個別ロスを取得
            loss = outputs.loss
            lm_loss = outputs.lm_loss
            n_step_loss = outputs.n_step_loss

            # n_step_loss に重みをかけたい場合、モデル内部ではなくここで合算してもOK
            # loss = lm_loss + n_step_loss_weight * n_step_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_n_step_loss += n_step_loss.item()

            global_step += 1

            if global_step % log_steps == 0:
                wandb.log(
                    {
                        "Train/Step Total Loss": loss.item(),
                        "Train/Step LM Loss": lm_loss.item(),
                        "Train/Step N-step Loss": n_step_loss.item(),
                        "Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            # eval & save
            if global_step % eval_and_save_steps == 0 and global_step > 0:
                avg_eval_loss = evaluate(
                    model, eval_loader, device, save_path, global_step
                )
                print(f"[Eval] Step {global_step}, Loss: {avg_eval_loss:.4f}")

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    model.save_pretrained(os.path.join(save_path, f"step_{global_step}"))
                    print(f"Best model saved at step {global_step}")

        avg_epoch_loss = total_loss / len(train_loader)
        avg_epoch_lm_loss = total_lm_loss / len(train_loader)
        avg_epoch_n_step_loss = total_n_step_loss / len(train_loader)

        wandb.log(
            {
                "Train/Epoch Total Loss": avg_epoch_loss,
                "Train/Epoch LM Loss": avg_epoch_lm_loss,
                "Train/Epoch N-step Loss": avg_epoch_n_step_loss,
                "Train/Epoch": epoch,
            },
            step=global_step,
        )

        print(
            f"Epoch {epoch} done. Total Loss: {avg_epoch_loss:.4f}, LM: {avg_epoch_lm_loss:.4f}, N-step: {avg_epoch_n_step_loss:.4f}"
        )


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
