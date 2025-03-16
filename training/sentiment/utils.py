import torch
import torch.optim as optim
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


class TextWithTokenSentimentDataset(Dataset):
    def __init__(self, texts, word_sentiments, tokenizer, max_length=32):
        """
        texts: list of strings (元のテキスト)
        word_sentiments: list of lists (単語ごとの感情ラベル)
            e.g. word_sentiments[i] = [0,1,2] -> i番目のテキストに対応する単語ごとの感情ラベル
        tokenizer: トークナイザ
        max_length: 最大長（パディングや切り詰め処理に使用）
        """
        self.texts = texts
        self.word_sentiments = word_sentiments  # 各単語ごとの感情ラベル
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.word_sentiments[
            idx
        ]  # 例: [0, 0, 1] のような単語単位の感情ラベル

        # 1. トークナイズ (subword単位に分割)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,  # 単語単位での位置情報を取得
        )
        input_ids = encoding["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # (seq_len, 2)

        seq_len = input_ids.size(0)  # トークン数

        # 2. 感情ラベルのトークン単位への拡張
        token_sentiments = torch.ones(seq_len, dtype=torch.long) * (
            -100
        )  # 初期値はignore_index (-100)
        word_index = 0  # 単語ラベルのインデックス

        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end != 0:  # 新しい単語の開始（通常、最初のサブワード）
                if word_index < len(word_labels):
                    token_sentiments[token_idx] = word_labels[
                        word_index
                    ]  # 単語の感情を適用
                    word_index += 1  # 次の単語へ
            else:
                # サブワードトークンは前のトークンのラベルをコピー
                if token_idx > 0:
                    token_sentiments[token_idx] = token_sentiments[token_idx - 1]

        # 3. LM 用のラベル
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,  # 言語モデル用 (seq_len,)
            "branch_labels": token_sentiments,  # 感情分類用 (seq_len,)
        }


def get_dataloader(tokenizer, batch_size=2, max_length=2048):
    train_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="train")
    test_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="test")

    train_texts = train_dataset["text"]
    train_token_sentiments = [[s] for s in train_dataset["label"]]

    test_texts = test_dataset["text"]
    test_token_sentiments = [[s] for s in test_dataset["label"]]

    # データセット & データローダ
    train_dataset = TextWithTokenSentimentDataset(
        texts=train_texts,
        word_sentiments=train_token_sentiments,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TextWithTokenSentimentDataset(
        texts=test_texts,
        word_sentiments=test_token_sentiments,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(
    model,
    train_loader,
    eval_loader,
    save_path="./",
    log_steps=100,
    eval_and_save_steps=500,
    num_epochs=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    best_eval_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_step = 0

    num_epochs = 5
    for _ in trange(num_epochs):
        model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_branch_loss = 0.0

        for batch in tqdm(train_loader):
            # バッチをGPUへ
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # LM用
            branch_labels = batch["branch_labels"].to(device)  # 感情分類用

            # 勾配クリア
            optimizer.zero_grad()

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                branch_labels=branch_labels,
                return_dict=True,
            )
            loss = outputs.loss

            loss_lm = (
                outputs.loss_backbone
                if hasattr(outputs, "loss_backbone")
                else torch.tensor(0.0)
            )
            loss_branch = (
                outputs.loss_branch
                if hasattr(outputs, "loss_branch")
                else torch.tensor(0.0)
            )

            # Backward & update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_lm_loss += loss_lm.item()
            total_branch_loss += loss_branch.item()

            global_step += 1

            # 📝 log_steps ごとに wandb に記録
            if global_step % log_steps == 0:
                wandb.log(
                    {
                        "Train/Step Loss": loss.item(),
                        "Train/Step LM Loss": loss_lm.item(),
                        "Train/Step Branch Loss": loss_branch.item(),
                    }
                )

            # 📌 eval_and_save_steps ごとに評価と保存
            if global_step % eval_and_save_steps == 0 and global_step > 0:
                avg_eval_loss = evaluate(
                    model, eval_loader, device, save_path, global_step
                )
                print(f"📊 Eval Step {global_step}, Loss: {avg_eval_loss:.4f}")

                # モデルを保存（Eval Loss が最小の場合のみ保存）
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    torch.save(model.state_dict(), save_path)
                    print(
                        f"✅ Model saved at step {global_step} with Eval Loss {best_eval_loss:.4f}"
                    )


def evaluate(model, eval_loader, device, save_path, global_step):
    model.eval()
    total_eval_loss = 0.0
    total_eval_lm_loss = 0.0
    total_eval_branch_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            branch_labels = batch["branch_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                branch_labels=branch_labels,
                return_dict=True,
            )
            loss = outputs.loss
            loss_lm = (
                outputs.loss_backbone
                if hasattr(outputs, "loss_backbone")
                else torch.tensor(0.0)
            )
            loss_branch = (
                outputs.loss_branch
                if hasattr(outputs, "loss_branch")
                else torch.tensor(0.0)
            )

            total_eval_loss += loss.item()
            total_eval_lm_loss += loss_lm.item()
            total_eval_branch_loss += loss_branch.item()

            # 感情分類の予測
            branch_logits = outputs.branch_logits  # (batch, seq_len, 3)
            preds = torch.argmax(branch_logits, dim=-1)  # (batch, seq_len)
            mask = branch_labels != -100  # パディング部分を無視
            preds = preds[mask]
            labels = branch_labels[mask]

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_eval_loss = total_eval_loss / len(eval_loader)
    avg_eval_lm_loss = total_eval_lm_loss / len(eval_loader)
    avg_eval_branch_loss = total_eval_branch_loss / len(eval_loader)

    # メトリクス計算
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(
        f"📊 Eval Results - Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}"
    )

    wandb.log(
        {
            "Eval/Loss": avg_eval_loss,
            "Eval/LM Loss": avg_eval_lm_loss,
            "Eval/Branch Loss": avg_eval_branch_loss,
            "Eval/Accuracy": accuracy,
            "Eval/Precision": precision,
            "Eval/Recall": recall,
            "Eval/F1-score": f1,
        }
    )

    model.save_pretrained(save_path + f"step_{global_step}")

    return avg_eval_loss
