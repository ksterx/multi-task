if __name__ == "__main__":
    from transformers import AutoTokenizer, Gemma3ForCausalLM
    from utils import get_dataloader, train

    from nn.branch import Gemma3WithBranch

    model_name = "google/gemma-3-1b-it"
    model = Gemma3WithBranch.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader, eval_loader = get_dataloader(tokenizer, batch_size=2, max_length=2048)

    train(
        model,
        train_loader,
        eval_loader,
        save_path="./",
        log_steps=100,
        eval_and_save_steps=500,
        num_epochs=10,
    )
