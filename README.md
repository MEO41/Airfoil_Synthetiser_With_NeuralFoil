
# ✈️ Synthetic-Airfoil-Factory

A **research-grade yet runnable pipeline** that learns from 1,647 real airfoil geometries,  
conditions on aerodynamic targets *(Re, α, C<sub>L</sub>, C<sub>D</sub>, C<sub>M</sub>, C<sub>L</sub>/C<sub>D</sub>)*, and generates brand-new 100-point `.dat` shapes.  
Quality is verified on-the-fly with **NeuralFoil**.
![Pipeline](https://github.com/MEO41/Synthetic_Airfoil_Generatorv2/blob/main/assets/pipline.png?raw=true)
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


**Hardware**: a single RTX 4090 trains the β-CVAE (~65k it/s, 40 mins, 200 epochs).  
CPU-only mode works but is ~30× slower.

![Outputs](https://raw.githubusercontent.com/MEO41/Airfoil_Synthetiser_With_NeuralFoil/914d40d79efaf8e0e90cf74ec1ecea02a1547d31/assets/example.svg)

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

## 🧪 Dependencies

| Package                                | Version Tested |
|----------------------------------------|----------------|
| PyTorch                                | 2.3            |
| PyTorch-Lightning                      | 2.2            |
| NeuralFoil                             | 0.6            |
| pandas / numpy                         | latest         |
| tqdm, joblib, scikit-learn, matplotlib |                |


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
