import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import os


# ----------------------------------------------------
# Dataset & DataLoader
# ----------------------------------------------------
class TextForMultiHeadCausalLMDataset(Dataset):
    """
    Returns a dictionary of:
      - input_ids
      - attention_mask
      - labels (standard LM training)
      - n_step_labels (for the 'n-step ahead' head)
    For demonstration, we define n_step_labels as a shifted version of the same text.
    """

    def __init__(self, texts, tokenizer, max_length=512, n_step=2):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize & pad
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Standard LM: labels are just input_ids (shifted internally by HF)
        labels = input_ids.clone()

        # n-step labels: shift the sequence by `n_step` positions
        # Example: if n_step=2, then each token's "label" is 2 tokens in the future
        # We'll set -100 for out-of-range positions so they don't contribute to the loss
        n_step_labels = input_ids.clone()
        n_step_labels[: -self.n_step] = n_step_labels[
            self.n_step :
        ].clone()  # shift forward
        # The last n_step positions can't predict anything valid, so set them to -100
        if self.n_step > 0:
            n_step_labels[-self.n_step :] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_step_labels": n_step_labels,
        }


def get_dataloader(
    tokenizer, batch_size=2, max_length=32, n_step=2, test_size=100, shuffle_train=True
):
    """
    A demonstration data loader. In real usage, load your own dataset or
    pass in a local path. We'll use "facebook/natural_reasoning" as an example.
    """

    def apply_chat_format(example):
        return {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["reference_answer"]},
                ],
                tokenize=False,
            )
        }

    dataset = load_dataset("facebook/natural_reasoning", split="train").map(
        apply_chat_format
    )
    dataset = dataset.train_test_split(test_size=100)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_texts = train_dataset["text"]
    test_texts = test_dataset["text"]

    train_dataset = TextForMultiHeadCausalLMDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        n_step=n_step,
    )
    test_dataset = TextForMultiHeadCausalLMDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        n_step=n_step,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ----------------------------------------------------
# Training & Evaluation loops
# ----------------------------------------------------
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
    Model training loop.
    """
    os.makedirs(save_path, exist_ok=True)
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project="farsight", name="multi-head-training")

    global_step = 0
    progress_bar = trange(num_epochs, desc="Epoch")

    for epoch in progress_bar:
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Train Iter", leave=False)

        # Accumulate losses to log
        accumulated_loss = 0
        accumulated_loss_head1 = 0
        accumulated_loss_head2 = 0

        for step, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)

            accumulated_loss += outputs.loss.detach()
            accumulated_loss_head1 += outputs.loss_head1.detach()
            accumulated_loss_head2 += outputs.loss_head2.detach()

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if (global_step + 1) % log_steps == 0:
                    avg_loss = (
                        accumulated_loss / accelerator.gradient_accumulation_steps
                    )
                    avg_loss_head1 = (
                        accumulated_loss_head1 / accelerator.gradient_accumulation_steps
                    )
                    avg_loss_head2 = (
                        accumulated_loss_head2 / accelerator.gradient_accumulation_steps
                    )

                    # On multi-GPU, gather and then average
                    avg_loss = accelerator.gather(avg_loss).mean()
                    avg_loss_head1 = accelerator.gather(avg_loss_head1).mean()
                    avg_loss_head2 = accelerator.gather(avg_loss_head2).mean()

                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "train/loss": avg_loss.item(),
                                "train/loss_head1": avg_loss_head1.item(),
                                "train/loss_head2": avg_loss_head2.item(),
                            },
                            step=global_step,
                        )

                    # Reset accumulators
                    accumulated_loss = 0.0
                    accumulated_loss_head1 = 0.0
                    accumulated_loss_head2 = 0.0

            if global_step % eval_and_save_steps == 0:
                evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    accelerator=accelerator,
                    save_path=save_path,
                    global_step=global_step,
                )
                # Save the entire state (model, optimizer, rng, etc.)
                if accelerator.is_main_process:
                    accelerator.save_state(
                        os.path.join(save_path, f"checkpoint-{global_step}")
                    )

    if accelerator.is_main_process:
        wandb.finish()


def evaluate(model, eval_loader, accelerator, save_path, global_step):
    model.eval()
    total_eval_loss = 0.0
    total_eval_lm_loss = 0.0
    total_eval_n_step_loss = 0.0
    total = 0

    device = accelerator.device

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
    print(
        f"[Eval] Global Step: {global_step}, "
        f"Loss: {avg_eval_loss:.4f}, "
        f"LM Loss: {avg_eval_lm_loss:.4f}, "
        f"N-step Loss: {avg_eval_n_step_loss:.4f}, "
    )

    wandb.log(
        {
            "Eval/Loss": avg_eval_loss,
            "Eval/LM Loss": avg_eval_lm_loss,
            "Eval/N-step Loss": avg_eval_n_step_loss,
        },
        step=global_step,
    )

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(save_path, f"step_{global_step}"))

    return avg_eval_loss
