if __name__ == "__main__":
    from transformers import AutoTokenizer
    from utils import get_dataloader, train
    import torch
    from nn.branch import Gemma2ForMultiHeadCausalLM
    from accelerate import Accelerator
    from accelerate.utils import DummyOptim, DummyScheduler

    # Acceleratorの初期化（DeepSpeed設定を使用）
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        # mixed_precision="bf16",
    )

    # デバイスの確認
    print(f"Using device: {accelerator.device}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")

    model_name = "google/gemma-2-9b-it"
    model = Gemma2ForMultiHeadCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False  # 訓練時はキャッシュを無効化
    )

    # メモリ効率化のための設定
    model.gradient_checkpointing_enable()  # gradient checkpointingを有効化

    if accelerator.is_main_process:
        # print trainable parameters
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Head 1 parameters: {sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)}")
        print(f"Head 2 parameters: {sum(p.numel() for p in model.lm_head2.parameters() if p.requires_grad)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # バッチサイズを小さく設定
    train_loader, eval_loader = get_dataloader(tokenizer, batch_size=8, max_length=1024)

    # オプティマイザの設定
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.999),
    )

    # Acceleratorで必要なコンポーネントを準備
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    train(
        model,
        train_loader,
        eval_loader,
        accelerator=accelerator,
        optimizer=optimizer,
        save_path="/lustre/k_ishikawa/results/farsight",
        log_steps=10,
        eval_and_save_steps=2000,
        num_epochs=1,
    )
