import argparse

import torch
from transformers import AutoTokenizer

from frac_ssm import FracForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a frac-ssm checkpoint.")
    parser.add_argument("model", help="Local checkpoint directory or Hugging Face model ID.")
    parser.add_argument("--tokenizer", help="Tokenizer path or model ID. Defaults to the checkpoint.")
    parser.add_argument("--prompt", default="The fractional state space model")
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument(
        "--device",
        help="Optional PyTorch device. By default, use PyTorch's default device.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = FracForCausalLM.from_pretrained(args.model).to(dtype=dtype)
    if args.device is not None:
        model = model.to(args.device)
    model.eval()

    model_device = next(model.parameters()).device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model_device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    prompt_length = inputs["input_ids"].shape[1]
    print(tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
