# Machine Learning Playground
Small, educational ML examples.

Files
- `MPNeuron.py` — minimal perceptron-like neuron (NumPy). Supports 1D/2D weights and inputs; raises `ValueError` on shape mismatch.

Quick start
- Install (pip): `pip install -r requirements.txt`
- Or run with `uv` (runs command inside `uv` environment):

	```bash
	uv run python MPNeuron.py
	# or install deps inside uv: uv run pip install -r requirements.txt
	```

Output
- Example interactive run prints numeric outputs and a thresholded result; a simple non-interactive example prints 
``` 
[[ 6]
 [12]]
```

More examples will be added.

