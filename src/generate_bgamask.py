import cv2
import numpy as np
import argparse
import os

def convert_values(img):
    """Converte valores 1 → 0 e 2 → 255"""
    out = img.copy()
    out[img == 1] = 0
    out[img == 2] = 255
    return out

def process_all_folders(root_dir, input_filename="segmentation_00000.png", output_filename="bgamask_00000.png"):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if input_filename in filenames:
            input_path = os.path.join(dirpath, input_filename)
            output_path = os.path.join(dirpath, output_filename)

            # Carrega a imagem
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"[AVISO] Não foi possível ler: {input_path}")
                continue

            # Garante que é imagem de um canal
            if len(img.shape) != 2:
                print(f"[AVISO] Imagem não é de um canal: {input_path}")
                continue

            # Converte e salva
            converted = convert_values(img)
            cv2.imwrite(output_path, converted.astype(np.uint8))
            print(f"[OK] Processado: {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho da imagem de entrada")
    # parser.add_argument("--output", required=True, help="Caminho da imagem de saída")
    args = parser.parse_args()

    process_all_folders(args.input)
