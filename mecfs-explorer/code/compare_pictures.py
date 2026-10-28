from pathlib import Path
from PIL import Image


folders = [
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output/top_n_variable_vst_genes_all_patients/clustering_plots_5pcs_1000genes_6clusters/"),
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output_old/clustering_plots_5pcs_1000genes_6clusters/"),
]

folders = [
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output/top_n_variable_vst_genes_all_patients/clustering_plots_5pcs_2000genes_7clusters/"),
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output_old/clustering_plots_5pcs_2000genes_7clusters/"),
]

folders = [
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output/top_n_variable_vst_genes_all_patients/clustering_plots_5pcs_5000genes_7clusters/"),
    Path("/home/jana/Documents/Projekte/MECFS_EDA_ML/mecfs-explorer/output_old/clustering_plots_5pcs_5000genes_7clusters/"),
]

image_extensions = {".png"}

current = {
    p.name: p for p in folders[0].iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
}
old = {
    p.name: p for p in folders[1].iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
}

for name in sorted(current.keys() | old.keys()):
    if name not in current:
        print(f"{name}: only in {folders[1]}")
        continue
    if name not in old:
        print(f"{name}: only in {folders[0]}")
        continue

    same_bytes = current[name].read_bytes() == old[name].read_bytes()

    with Image.open(current[name]) as img1, Image.open(old[name]) as img2:
        pixels1 = img1.convert("RGBA")
        pixels2 = img2.convert("RGBA")
        same_pixels = pixels1.size == pixels2.size and pixels1.tobytes() == pixels2.tobytes()

    print(f"{name}: same file={same_bytes}, same pixels={same_pixels}")