
# ✈️ Synthetic-Airfoil-Factory

A **research-grade yet runnable pipeline** that learns from 1,647 real airfoil geometries,  
conditions on aerodynamic targets *(Re, α, C<sub>L</sub>, C<sub>D</sub>, C<sub>M</sub>, C<sub>L</sub>/C<sub>D</sub>)*, and generates brand-new 100-point `.dat` shapes.  
Quality is verified on-the-fly with **NeuralFoil**.

| Stage                      | Notebook Cell              | What It Does |
|---------------------------|----------------------------|--------------|
| **1. Data Spine**         | `step-1-load-resample.ipynb` | Loads every `.dat`, resamples to 100 points, builds `df_geo.pkl`. |
| **2. Merge CSV**          | `step-2-merge-csv.ipynb`     | Joins aero table → `airfoil_cvae_dataset.pkl`. |
| **3. Normalize + DataLoader** | `step-3-dataloader.ipynb` | Global Min-Max scalers + PyTorch `Dataset` / `DataLoader`. |
| **4. Train β-CVAE**       | `step-4-train_cvae.ipynb`    | Convolutional β-CVAE *(geo branch + aero regressor)* with early-stopping. |
| **5. Inference & DDPM (Optional)** | `step-5-inference_ddpm.ipynb` | • ID-agnostic sampling  
• Saves top-k `.dat` files under `generated_airfoils/`  
• Optional latent DDPM for diversity. |
| **6. NeuralFoil Sweep**   | `step-6-neuralfoil.ipynb`    | Batches each synthetic airfoil through NeuralFoil (GPU-aware). |
| **7. Scoring vs Target**  | `step-7-compare_target.ipynb`| Computes absolute & % errors, weighted score table. |

---

## ⚡ Quick Start (One-GPU Demo)

```bash
git clone https://github.com/your-handle/Synthetic-Airfoil-Factory.git
cd Synthetic-Airfoil-Factory
conda env create -f env.yml   # or pip install -r requirements.txt
conda activate airfoil-factory
jupyter lab            # open the notebooks in order 1 → 7
```

**Hardware**: a single RTX 4090 trains the β-CVAE (~65k it/s, 40 mins, 200 epochs).  
CPU-only mode works but is ~30× slower.

---

## 📁 Folder Layout

```
│  README.md
│  env.yml
├─airfoils/                 # 1,647 original *.dat
├─data/
│   ├─df_geo.pkl            # step-1 output
│   └─airfoil_cvae_dataset.pkl
├─checkpoints/
│   └─beta-cvae-*.ckpt      # best models (Lightning)
├─generated_airfoils/       # step-5 outputs (.dat)
├─neuralfoil_results/       # step-6 CSVs
└─notebooks/
    ├─step-1-load-resample.ipynb
    ├─step-2-merge-csv.ipynb
    └─…
```

---

## 🧠 Key Algorithms

| Block                        | Design Notes                                                                                                                       |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **β-CVAE**                   | 1-D conv encoder, 32-D latent, geometry MSE + β·KLD + λ·aero-MSE.                                                                  |
| **Condition Vector**         | 6 normalized numerics + 1,647-way one-hot ID.                                                                                      |
| **ID-Agnostic Generation**   | During sampling, the one-hot slice is set to **zeros** (or random) so the model can’t just memorize a training shape.              |
| **Latent DDPM (Optional)**   | DDPM learns a distribution *inside* the 32-D latent → easy diversity boost without retraining the decoder.                         |
| **Validation**               | NeuralFoil yields C<sub>L</sub>, C<sub>D</sub>, C<sub>M</sub> in ~2 ms; candidates are ranked by a weighted absolute-error score. |

---

## 🔁 Re-Using the Model

```python
from model import BetaCVAE, geom_scaler, cond_scaler
spec = dict(Re=1e6, alpha=8, cl=1.0, cd=0.015, cm=-0.05, cl_cd_ratio=67)
airfoils = sample_airfoils(spec, n_samples=200, top_k=10)  # returns list of (100,2) numpy arrays
```

---

## 🧪 Dependencies

| Package                                | Version Tested |
|----------------------------------------|----------------|
| PyTorch                                | 2.3            |
| PyTorch-Lightning                      | 2.2            |
| NeuralFoil                             | 0.6            |
| pandas / numpy                         | latest         |
| tqdm, joblib, scikit-learn, matplotlib |                |

See `env.yml` for full environment.

---

## 📝 Citation / Credits

- **NeuralFoil**: T. Flack & O. Stanford, 2024  
- Original airfoil geometries collected from UIUC and AirfoilTools  
- β-VAE formulation inspired by *Higgins et al., ICLR 2017*

---

## 🪪 License

MIT for code.  
Generated airfoils are released under **CC-BY 4.0** — feel free to use them, citing this repo.

---

*Happy synthesizing!* 🚀✈️
