if __name__ == "__main__":
    from transformers import AutoTokenizer
    from utils import get_dataloader, train

    from nn.branch import Gemma3WithBranch

    model_name = "google/gemma-3-1b-it"
    model = Gemma3WithBranch.from_pretrained(
        model_name, branch_hidden_size=512, dropout_rate=0.3
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader, eval_loader = get_dataloader(tokenizer, batch_size=8, max_length=2048)

    train(
        model,
        train_loader,
        eval_loader,
        save_path="/lustre/k_ishikawa/results/sentiment",
        log_steps=10,
        eval_and_save_steps=1000,
        num_epochs=1,
        lr=1e-5,
    )
