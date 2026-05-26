# Progetto: ForestFormer3D × TreeScanPL10K

## Stato attuale (aggiornato: maggio 2026)

| Fase | Status |
|------|--------|
| Repo FF3D clonato | ✅ DONE |
| Ambiente configurato | ✅ DONE |
| Inference testata su 1 plot TreeScanPL10K | ✅ DONE |
| Zero-shot eval completa (tutti i plot) | ⏳ PROSSIMO |
| Fine-tuning + species head | 🔜 |
| Valutazione finale + ablation | 🔜 |
| Writing | 🔜 |

> **Nota:** I risultati dell'inference iniziale su un singolo plot sono da documentare e usare come punto di partenza per capire il domain gap ULS→TLS.

---

## Obiettivo del progetto

Estendere **ForestFormer3D** con una **species classification head** e validarlo su **TreeScanPL10K**.
Deadline: **31 luglio 2026**.

Contributo scientifico: primo benchmark di metodi end-to-end di instance segmentation
su TreeScanPL10K (dataset TLS appena pubblicato, nessun baseline esistente), con estensione
per species classification — task non presente nel paper originale ForestFormer3D.

---

## ForestFormer3D — architettura

- **Paper:** arXiv 2506.16991v2
- **Repo + pesi pretrained:** https://bxiang233.github.io/FF3D/
- **Training originale:** FOR-instanceV2 (ULS airborne LiDAR)
- **SOTA originale:** F1 82.8%, Coverage 81.2%, mIoU 86.2%

```
Input XYZ (N×3)
  → Voxelizzazione 0.2m → Sparse tensor V (M×3)
  → 3D Sparse U-Net (SpConvUNet) → F (M×32)
  → MLP discriminativo → embedding 5D (M×5)   } ISA-guided
  → MLP classificatore → tree/non-tree (M×2)  } query selection
  → FPS nello spazio 5D → K_ins=300 query points
  → Transformer decoder × 6 (self-attn + cross-attn su F)
  → K_ins maschere istanza + K_sem=3 maschere semantiche + score K_ins
```

**Loss:**
```
L = L_total + L_binary + L_disc
L_total = Σ_{layer=1}^{6} (L_bce + L_dice + 0.5·L_score + 0.2·L_sem)
```

**Componenti chiave da capire nel codice:**
- `model/forestformer3d.py` — architettura principale
- `model/decoder.py` — transformer decoder con query points
- `utils/isa_sampling.py` — ISA-guided query point selection
- `utils/block_merging.py` — score-based NMS per inference
- `train.py` / `inference.py` — entry points

---

## Dataset: TreeScanPL10K

- **Paper:** Scientific Data 2026, doi:10.1038/s41597-026-07269-1
- **Zenodo:** https://zenodo.org/records/19127709
- **272 plot** circolari, raggio 15m, TLS (FARO Focus 3D X130 / Trimble TX5)
- **10.417 alberi**, 7.465 con species label (71.7%), 30 specie

### Struttura LAZ (campi per punto)

```
X, Y, Z          → coordinate 3D
intensity        → intensità laser
treeID           → instance label (0 = non-albero)
treeSP           → species index (0 = non-albero / specie sconosciuta)
completelyInside → 1 se corona non troncata dal bordo plot
```

### Specie principali

| Specie | N | Specie | N |
|--------|---|--------|---|
| Pinus sylvestris | 3833 | Carpinus betulus | 140 |
| Picea abies | 936 | Larix decidua | 117 |
| Fagus sylvatica | 597 | Alnus glutinosa | 88 |
| Quercus sp. | 530 | Tilia cordata | 84 |
| Abies alba | 453 | Quercus rubra | 54 |
| Betula pendula | 439 | Acer pseudoplatanus | 47 |

**Class imbalance:** specie con <40 individui → classe "other". Target: ~12 classi.

### Split geografico (NO random split — evita leakage spaziale)

```
Train: Milicz, Pieńsk, Herby, Dojlidy   (~180 plot)
Val:   Supraśl                           (~35 plot)
Test:  Gorlice                           (~62 plot)
```

### Metadati CSV

- `plot_summary.csv` — district, n_trees, species_composition per plot
- `ind_tree_summary.csv` — height_m, crown_area_m2, point_count per albero

---

## Decisioni tecniche già prese

### 1. Semantic branch: disabilitato

TreeScanPL10K non ha label ground/wood/leaf. Soluzione: `K_sem = 0` o
`use_semantic = False`. Focus su instance segmentation + species classification.
Niente L_sem nella loss.

### 2. Species classification head

```python
# Dopo il transformer decoder:
# F: (M, 32) — feature tensor
# instance_masks: (K_ins, M) — maschere binarie predette

# Masked mean pooling
instance_features = (instance_masks.unsqueeze(-1) * F.unsqueeze(0)).sum(1)  # (K_ins, 32)
instance_features = instance_features / (instance_masks.sum(1, keepdim=True) + 1e-6)

# Species MLP
species_logits = species_head(instance_features)  # (K_ins, n_species)

# Loss (solo su alberi con treeSP > 0)
L_species = F.cross_entropy(species_logits[valid_mask], species_gt[valid_mask],
                             weight=class_weights)

# Loss totale aggiornata
L = L_total + L_binary + L_disc + lambda_species * L_species
# lambda_species = 0.5 (da ablare)
```

### 3. Compatibilità plot → cylinder

Ogni plot TreeScanPL10K (r=15m) ≈ cylinder ForestFormer3D (r=16m).
Ogni plot può essere processato come un singolo blocco durante inference.
Nessun sliding window necessario (plot già piccoli).

### 4. Voxel resolution

TLS è molto più denso di ULS. Se si hanno OOM o tempi eccessivi:
- Prova `voxel_size = 0.3m` o `0.5m` invece di `0.2m`
- Oppure subsample a 500 pts/m² prima del voxeling

### 5. treeSP = 0 nella loss

Punti con `treeSP = 0` (unknown / non-albero) → escludi dalla species loss con mask:
```python
valid_mask = species_gt > 0
```

---

## Metriche target

| Task | Metriche |
|------|---------|
| Instance segmentation | Precision, Recall, F1 (IoU≥0.5), Coverage |
| Species classification | Per-class accuracy, Macro-F1, Confusion matrix |
| Confronto baseline | FF3D zero-shot vs FF3D fine-tuned vs TreeLearn (nativo TLS) |

---

## Prossimi task immediati

### ▶ Task corrente: zero-shot evaluation completa

FF3D è già stato testato su 1 plot. Ora: eseguire inference su **tutti i plot del test set**
(Gorlice, ~62 plot) con i pesi pretrained, senza fine-tuning.

```bash
# Per ogni plot nel test set:
python inference.py \
  --checkpoint pretrained_weights.pth \
  --input path/to/gorlice_plots/ \
  --output results/zero_shot/ \
  --no_semantic

# Poi calcola metriche aggregando tutti i plot:
python evaluate.py \
  --predictions results/zero_shot/ \
  --ground_truth path/to/gorlice_plots/ \
  --metric f1 coverage precision recall
```

**Output atteso:** tabella con P/R/F1/Coverage media su ~62 plot.
Confronta con i risultati di LAUTx (MLS) dal paper FF3D: F1 90.5% — il TLS
sarà probabilmente peggio, ed è questo il domain gap da documentare.

### Task successivo: data loader per TreeScanPL10K

Creare un dataset class PyTorch che:
1. Legge LAZ con `laspy`
2. Estrae XYZ (normalizzati al centroide del plot)
3. Restituisce `instance_labels` (da `treeID`) e `species_labels` (da `treeSP`)
4. Applica augmentation (flip, rotation, scaling come in FF3D)
5. Filtra punti con `completelyInside=0` se necessario

```python
class TreeScanPL10KDataset(torch.utils.data.Dataset):
    def __init__(self, laz_dir, split='train', voxel_size=0.2):
        # split definito per distretto geografico (vedi sopra)
        ...
    def __getitem__(self, idx):
        # ritorna: xyz (N,3), instance_labels (N,), species_labels (N,)
        ...
```

---

## Note tecniche critiche

**ULS vs TLS — il domain gap:**
- ULS: top-down, denso in chioma, sparso in fusto
- TLS: bottom-up, densissimo in fusto (occlusioni in chioma)
- ISA-guided selection funziona su voxel "tree" — in TLS il fusto è molto rappresentato
  mentre la chioma può essere sparsa → il 5D embedding potrebbe comportarsi diversamente

**Memoria GPU:**
- FF3D originale: A100 80GB, batch size 2
- TLS più denso → rischio OOM; ridurre batch size a 1 o aumentare voxel size

**TreeLearn come baseline TLS:**
- Progettato per TLS ground-based (stesso dominio di TreeScanPL10K)
- Repo: https://github.com/ecker-lab/TreeLearn
- Usarlo come upper bound / confronto principale

**Attenzione alla normalizzazione delle coordinate:**
- FF3D usa coordinate relative al blocco cilindrico
- TreeScanPL10K usa coordinate assolute EPSG 2180
- Normalizzare XYZ al centroide del plot prima di passarle al modello

---

## Riferimenti

| Risorsa | Link |
|---------|------|
| FF3D repo e pesi | https://bxiang233.github.io/FF3D/ |
| FF3D paper | arXiv 2506.16991 |
| TreeScanPL10K Zenodo | https://zenodo.org/records/19127709 |
| TreeScanPL10K paper | doi:10.1038/s41597-026-07269-1 |
| TreeLearn repo | https://github.com/ecker-lab/TreeLearn |
| Pre-segmentation code | https://github.com/maxkulicki/tree_presegmentation |
