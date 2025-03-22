if __name__ == "__main__":
    from transformers import AutoTokenizer
    from utils import get_dataloader, train

    from nn.branch import Gemma2ForMultiHeadCausalLM

    model_name = "google/gemma-2-9b-it"
    model = Gemma2ForMultiHeadCausalLM.from_pretrained(model_name)

    # print trainable parameters
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Head 1 parameters: {sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)}")
    print(f"Head 2 parameters: {sum(p.numel() for p in model.lm_head2.parameters() if p.requires_grad)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader, eval_loader = get_dataloader(tokenizer, batch_size=4, max_length=2048)

    train(
        model,
        train_loader,
        eval_loader,
        save_path="/lustre/k_ishikawa/results/farsight",
        log_steps=50,
        eval_and_save_steps=250,
        num_epochs=5,
        lr=5e-4,
    )
