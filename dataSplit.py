import yaml
import os
import shutil
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURAÇÃO
# -----------------------------
yaml_path = "./dataset/brain-tumor.yaml"
output_root = "dataset_split"

test_size = 0.20  # 20% test
val_size = 0.10   # 10% val


# -----------------------------
# FUNÇÃO PARA LISTAR PARES img/label
# -----------------------------
def load_pairs(images_dir, labels_dir):
    imgs = []
    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = os.path.join(images_dir, f)
            label = os.path.join(labels_dir, f.rsplit(".", 1)[0] + ".txt")
            imgs.append((img, label))
    return imgs


# -----------------------------
# LER YAML E RESOLVER CAMINHOS
# -----------------------------
with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

root_path = os.path.dirname(yaml_path)
dataset_root = os.path.join(root_path, data["path"])

train_imgs = os.path.join(dataset_root, data["train"])
train_lbls = train_imgs.replace("images", "labels")

val_imgs = os.path.join(dataset_root, data["val"])
val_lbls = val_imgs.replace("images", "labels")

# -----------------------------
# CARREGAR TODAS AS AMOSTRAS
# -----------------------------
pairs = []
pairs += load_pairs(train_imgs, train_lbls)
pairs += load_pairs(val_imgs, val_lbls)

print(f"Total de amostras encontradas: {len(pairs)}")


# -----------------------------
# SPLIT train/temp
# -----------------------------
train_pairs, temp_pairs = train_test_split(
    pairs,
    test_size=(val_size + test_size),
    random_state=42
)

# SPLIT temp em val/test
rel = test_size / (test_size + val_size)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=rel,
    random_state=42
)

print(f"Novo split: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")


# -----------------------------
# CRIAR PASTAS SAÍDA
# -----------------------------
for split in ["train", "val", "test"]:
    os.makedirs(f"{output_root}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_root}/labels/{split}", exist_ok=True)


# -----------------------------
# COPIAR ARQUIVOS
# -----------------------------
def copy_pairs(pairs, split):
    for img, lbl in pairs:
        shutil.copy(img, f"{output_root}/images/{split}/")
        if os.path.exists(lbl):
            shutil.copy(lbl, f"{output_root}/labels/{split}/")


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")


# -----------------------------
# GERAR NOVO YAML
# -----------------------------
new_yaml = {
    "path": output_root,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": data["names"]
}

with open(f"{output_root}/dataset.yaml", "w", encoding="utf-8") as f:
    yaml.dump(new_yaml, f)

print("Novo split criado em:", output_root)
