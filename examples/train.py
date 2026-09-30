import argparse

import torch

from frac_ssm import FracConfig, FracForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frac-ssm from scratch on synthetic tokens.")
    parser.add_argument("--config", default="configs/frac_1.3b.json")
    parser.add_argument("--output_dir")
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--sequence_length", default=512, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument(
        "--device",
        help="Optional PyTorch device. By default, use PyTorch's default device.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    config = FracConfig.from_pretrained(args.config, use_cache=False)
    model = FracForCausalLM(config).to(dtype=dtype)
    if args.device is not None:
        model = model.to(args.device)
    model.train()
    model_device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(args.steps):
        input_ids = torch.randint(
            config.vocab_size,
            (args.batch_size, args.sequence_length),
            device=model_device,
        )
        loss = model(input_ids=input_ids, labels=input_ids).loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f"step={step + 1} loss={loss.item():.6f}")

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
