from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io
import re
import sys
import json
import zipfile
from keras.layers import TFSMLayer

# === Inisialisasi Flask App ===
app = Flask(__name__)

# === Lokasi direktori aplikasi ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# === Daftar file zip model dari folder models/ dan tujuan ekstraksi ===
ZIP_MODELS = {
    "classifier": "models/model_ktp_classifier_savedmodel.zip",
    "ocr_general": "models/model_ocr_non_nik_savedmodel.zip",
    "ocr_nik": "models/model_ocr_nik_savedmodel.zip",
}


EXTRACT_PATHS = {
    "classifier": "/tmp/model_classifier",
    "ocr_general": "/tmp/model_ocr_general",
    "ocr_nik": "/tmp/model_ocr_nik",
}

# === Fungsi ekstrak zip jika belum ada ===
def extract_model(zip_filename, extract_to):
    zip_path = os.path.join(BASE_DIR, zip_filename)
    if not os.path.exists(extract_to):
        print(f"📦 Mengekstrak {zip_filename} ke {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Ekstraksi selesai: {zip_filename}")
    else:
        print(f"ℹ️ Sudah diekstrak sebelumnya: {extract_to}")

# === Ekstrak semua model ===
for key in ZIP_MODELS:
    extract_model(ZIP_MODELS[key], EXTRACT_PATHS[key])

# === Load semua model ===
try:
    print("🔄 Memuat model klasifikasi KTP...")
    model_classifier = TFSMLayer(EXTRACT_PATHS['classifier'], call_endpoint="serving_default")
    print("✅ model_ktp_classifier berhasil dimuat.")

    print("🔄 Memuat model OCR non-NIK...")
    model_ocr_general = TFSMLayer(EXTRACT_PATHS['ocr_general'], call_endpoint="serving_default")
    print("✅ ocr_non_nik_model berhasil dimuat.")

    print("🔄 Memuat model OCR NIK...")
    model_ocr_nik = TFSMLayer(EXTRACT_PATHS['ocr_nik'], call_endpoint="serving_default")
    print("✅ ocr_nik_model berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model_classifier = None
    model_ocr_general = None
    model_ocr_nik = None

def main(image_file):
    if not all([model_classifier, model_ocr_general, model_ocr_nik]):
        raise Exception("Model belum dimuat dengan benar.")

    IMG_SIZE = (224, 224)
    cropped_images = {}

    # --- Langkah 1: Baca dan prediksi apakah gambar adalah KTP ---
    img_pil = Image.open(image_file).convert('RGB')
    img_np = np.array(img_pil)
    img_for_pred = img_pil.resize(IMG_SIZE)
    img_array = image.img_to_array(img_for_pred) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_classifier.predict(img_array)[0][0]
    label = 'Bukan KTP' if prediction < 0.5 else 'KTP'

    if label != 'KTP':
        print("❌ Gambar yang diunggah bukan KTP.")
        return {}



    if label == 'KTP':
        image_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        def resize_image_with_fixed_dimensions(image, target_width=1720, target_height=906):
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        resized_image = resize_image_with_fixed_dimensions(image_cv)

        crop_left = int(resized_image.shape[1] * 0.23)
        crop_right = int(resized_image.shape[1] * 0.76) 
        crop_bottom = int(resized_image.shape[0] * 0.92)
        cropped_image = resized_image[:crop_bottom, crop_left:crop_right]
        cropped_images["input_image"] = cropped_image


    def apply_closing(image, kernel_size=(1, 11), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    closed_images = {}
    for filename, cropped_image in cropped_images.items():
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=45)
        gradient = cv2.morphologyEx(contrast, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dilated = cv2.dilate(otsu, np.ones((3, 3), np.uint8), iterations=1)
        closed = apply_closing(dilated, kernel_size=(32, 1), iterations=1)
        closed_images[filename] = closed

    cropped_lines = []
    for closed_image in closed_images.values():
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 35 and h > 15 and x < 300:
                cropped_lines.append(cropped_image[y:y+h, x:x+w])
    cropped_lines = cropped_lines[:11] if len(cropped_lines) == 16 else cropped_lines[:10] if len(cropped_lines) == 15 else cropped_lines

    def apply_closing(img, kernel_size=(5, 5), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    all_cropped_words_per_line = []
    for i, line in enumerate(cropped_lines):
        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        alpha, beta = (1.3, 50) if i != 1 and i != 2 else (1.3, 60)
        contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        kernel = np.ones((5, 5), np.uint8) if i != 1 and i != 2 else np.ones((3, 4), np.uint8)
        gradient = cv2.morphologyEx(contrast, cv2.MORPH_GRADIENT, kernel)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closing_kernel = (11, 1) if i == 0 else (16, 1) if i == 1 else (30, 2) if i == 2 else (12, 1)
        closed = apply_closing(binary, kernel_size=closing_kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        words = []
        for c in sorted_contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 25:
                word = line[y:y + h, x:x + w]
                words.append(word)
        if i == 0: words = words[1:]
        all_cropped_words_per_line.append(words)

    def unsharp_mask(image, sigma=0.5, strength=0.5):
        blur = cv2.GaussianBlur(image, (0, 0), sigma)
        return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

    cropped_characters_grouped = {}
    for line_idx, words_in_line in enumerate(all_cropped_words_per_line):
        cropped_characters_grouped[line_idx] = {}
        for word_idx, word in enumerate(words_in_line):
            gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            sharp = unsharp_mask(gray)
            denoised = cv2.fastNlMeansDenoising(sharp, None, h=5, templateWindowSize=7, searchWindowSize=21)
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
            contrast = cv2.convertScaleAbs(blurred, alpha=1.2, beta=43)
            thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            eroded = cv2.erode(thresh, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if line_idx < 2:
                min_width, min_height = 5, 20
                split_threshold_1 = 80
                split_threshold_2 = 60
                split_threshold_3 = None
            else:
                min_width, min_height = 3, 15
                split_threshold_1 = 80
                split_threshold_2 = 65
                split_threshold_3 = 45

            valid_boxes = [(x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours] if w > min_width and h > min_height]
            if not valid_boxes: continue
            max_height = max(h for _, _, _, h in valid_boxes)
            chars = []
            for x, y, w, h in valid_boxes:
                y_new = max(y - (max_height - h) // 2, 0)
                h_new = min(max_height, word.shape[0] - y_new)
                if split_threshold_3 and w > split_threshold_1:
                    for i in range(4):
                        chars.append((x + i * w // 4, word[y_new:y_new + h_new, x + i * w // 4:x + (i + 1) * w // 4]))
                elif w > split_threshold_2:
                    for i in range(3):
                        chars.append((x + i * w // 3, word[y_new:y_new + h_new, x + i * w // 3:x + (i + 1) * w // 3]))
                elif split_threshold_3 and w > split_threshold_3:
                    for i in range(2):
                        chars.append((x + i * w // 2, word[y_new:y_new + h_new, x + i * w // 2:x + (i + 1) * w // 2]))
                else:
                    chars.append((x, word[y_new:y_new + h_new, x:x + w]))
            chars = sorted(chars, key=lambda item: item[0])
            cropped_characters_grouped[line_idx][word_idx] = [char[1] for char in chars]

    def preprocess_character(image, target_size=(64, 64)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA) / 255.0
        return np.expand_dims(image, axis=(0, -1))

    def predict_character(image, model, label_map):
        pred = model.predict(preprocess_character(image))
        return label_map[np.argmax(pred)]

    def predict_line(chars, model, label_map):
        return " ".join("".join(predict_character(c, model, label_map) for c in word) for word in chars.values())

    label_map_general = {i: ch for i, ch in enumerate(
        [',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', '/', ' '])}
    label_map_nik = {i: str(i) for i in range(10)}

    
    


    predicted_lines = []
    for idx, chars in cropped_characters_grouped.items():
        model = model_ocr_nik if idx == 2 else model_ocr_general
        label_map = label_map_nik if idx == 2 else label_map_general
        predicted_lines.append(predict_line(chars, model, label_map))

    def correct_ocr_to_alpha(text):
        correction = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G', '@': 'A', '$': 'S', '&': 'E'}
        return re.sub(r'[^A-Z\s]', '', ''.join(correction.get(c.upper(), c) for c in text.upper()))

    def correct_common_ocr_errors(text):
        return ''.join({'O': '0', 'o': '0', 'I': '1', 'l': '1', 'Z': '2', 'S': '5', 'B': '8'}.get(c, c) for c in text)

    def split_birth_info(line):
        i = next((i for i, c in enumerate(line) if c.isdigit()), None)
        return (line[:i].strip(), line[i:].strip()) if i is not None else (line.strip(), "")

    def extract_birth_info(line):
        parts = re.findall(r'\w+', line)
        tempat = parts[0] if parts else ""
        digits = correct_common_ocr_errors("".join(parts[1:]))
        if digits.isdigit() and len(digits) in [6, 8]:
            return tempat.upper(), f"{digits[:2]}-{digits[2:4]}-{digits[4:]}"
        return tempat.upper(), " ".join(parts[1:])

    def process_jenis_kelamin(text):
        clean = re.sub(r'[^A-Z]', '', text.upper().replace(" ", ""))
        for p in [r'^LAKI[LAKI]*$', r'^LAK[ILAK]*$', r'^LK[ILAK]*$', r'^LA[KILA]*$', r'^L[AKIL]*$', r'LAK1', r'PRIA', r'LAKIL$']:
            if re.fullmatch(p, clean): return "LAKI-LAKI"
        return "PEREMPUAN"

    def fix_kota_kabupaten(text):
        text = re.sub(r'[^A-Z\s]', '', text.upper())
        text = re.sub(r'KAB(?:\s*UPATEN|\s+)', 'KABUPATEN ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if 'KABUPATEN' in text: return text
        if 'KOTA' in text: return text
        if 'KAB' in text: return re.sub(r'KAB\s+', 'KABUPATEN ', text)
        return text

    # Raw result
    raw_ktp_data = {
        "provinsi": predicted_lines[0] if len(predicted_lines) > 0 else "",
        "kota_kabupaten": predicted_lines[1] if len(predicted_lines) > 1 else "",
        "nik": predicted_lines[2] if len(predicted_lines) > 2 else "",
        "nama": predicted_lines[3] if len(predicted_lines) > 3 else "",
        "jenis_kelamin": predicted_lines[5] if len(predicted_lines) > 5 else "",
        "alamat": " ".join(predicted_lines[6:10]) if len(predicted_lines) >= 10 else "",
        "tempat_lahir & tanggal_lahir": predicted_lines[4] if len(predicted_lines) > 4 else ""
    }

    if len(predicted_lines) == 11:
        nama_lengkap = correct_ocr_to_alpha(predicted_lines[3] + " " + predicted_lines[4])
        if len(predicted_lines[5].split()) == 1:
            tempat_lahir, tanggal_lahir = split_birth_info(predicted_lines[5])
        else:
            tempat_lahir, tanggal_lahir = extract_birth_info(predicted_lines[5])

        processed_ktp_data = {
            "provinsi": correct_ocr_to_alpha(predicted_lines[0]),
            "kota_kabupaten": correct_ocr_to_alpha(fix_kota_kabupaten(predicted_lines[1])),
            "nik": predicted_lines[2],
            "nama": correct_ocr_to_alpha(nama_lengkap),
            "tempat_lahir": tempat_lahir,
            "tanggal_lahir": tanggal_lahir,
            "jenis_kelamin": process_jenis_kelamin(predicted_lines[6]),
            "alamat": " ".join(predicted_lines[7:11])
        }
    elif len(predicted_lines) == 10:
        if len(predicted_lines[4].split()) == 1:
            tempat_lahir, tanggal_lahir = split_birth_info(predicted_lines[4])
        else:
            tempat_lahir, tanggal_lahir = extract_birth_info(predicted_lines[4])

        processed_ktp_data = {
            "provinsi": correct_ocr_to_alpha(predicted_lines[0]),
            "kota_kabupaten": correct_ocr_to_alpha(fix_kota_kabupaten(predicted_lines[1])),
            "nik": predicted_lines[2],
            "nama": correct_ocr_to_alpha(predicted_lines[3]),
            "tempat_lahir": tempat_lahir,
            "tanggal_lahir": tanggal_lahir,
            "jenis_kelamin": process_jenis_kelamin(predicted_lines[5]),
            "alamat": " ".join(predicted_lines[6:10])
        }
    else:
        print("Jumlah baris tidak sesuai format KTP.")
        processed_ktp_data = {}

    return processed_ktp_data
    
# === API Endpoint ===
@app.route('/ocr-ktp', methods=['POST'])
def ocr_ktp():
    if 'image' not in request.files:
        return jsonify({"status": "failed", "message": "No image uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"status": "failed", "message": "No image selected"}), 400

    try:
        processed_ktp_data = main(image_file)
        return jsonify({
            "status": "success" if processed_ktp_data else "failed",
            "data": processed_ktp_data
        })
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
