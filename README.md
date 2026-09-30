# Fractional State Space Transition for Long Sequence Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2609.36314-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.36314)

This repository is the official implementation of the paper **Fractional State Space Transition for Long Sequence Modeling**, accepted to **NeurIPS 2026 (Oral)**. 


## Latest Updates

- **Coming soon** The efficient implementation of FRAC.
- **`09/29/2026`** A pure PyTorch hardware-agnostic implementation of FRAC that is mathematically identical to the description in the paper. It is intended for reference purposes only: **this is not the efficient implementation described in the paper.**
 
## Quick Start

The model follows the Hugging Face causal language-model interface.

```python
import torch
from transformers import AutoTokenizer

from frac_ssm import FracForCausalLM

checkpoint = "path/to/frac-ssm-checkpoint"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = FracForCausalLM.from_pretrained(checkpoint).eval()

inputs = tokenizer("The future of long-context modeling is", return_tensors="pt")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation

If you use FRAC in your work, please cite the following paper.

```bibtex
@article{kobyzev2026fractional,
  title   = {Fractional State Space Transition for Long Sequence Modeling},
  author  = {Kobyzev, Ivan and Ghaddar, Abbas and Nasiri-Sarvi, Ali and Shang, Lifeng and Cui, Yufei},
  journal = {arXiv preprint arXiv:2609.36314},
  year    = {2026}
}
```

## Contact

For questions about the code, please contact [Ali Nasiri-Sarvi](https://github.com/anasiri).
