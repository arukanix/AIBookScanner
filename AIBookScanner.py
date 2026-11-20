import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from tkinter import filedialog as fd
from tkinter import messagebox
import uuid
import zipfile

import customtkinter as ctk
import cv2
from docuwarp.unwarp import Unwarp
import img2pdf
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def find_rar_executable() -> str | None:
    """システム内の rar / WinRAR 実行ファイルを検索"""
    for cmd in ("rar", "Rar", "winrar", "WinRAR"):
        found = shutil.which(cmd)
        if found:
            return found
    common_paths = [
        r"C:\Program Files\WinRAR\Rar.exe",
        r"C:\Program Files (x86)\WinRAR\Rar.exe",
        r"C:\Program Files\WinRAR\WinRAR.exe",
        r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    return None


RAR_EXE = find_rar_executable()


# ==========================================
# 画像処理ロジック
# ==========================================
def natural_sort_key(s):
    """ファイル名を数値順（1, 2, 10...）に並べるためのキー"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def crop(image: np.ndarray, model: YOLO, device: str | int) -> np.ndarray:
    results = model(image, classes=[73], verbose=False, device=device)
    final_crop = image.copy()
    for result in results:
        if result.masks is None:
            continue
        for j, mask in enumerate(result.masks.xy):
            if j != 0:
                break
            points = np.array(mask, dtype=np.int32)
            h, w, _ = image.shape
            cropped_bgr = np.zeros_like(image)
            mask_layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_layer, [points], 255)
            cropped_bgr[mask_layer == 255] = image[mask_layer == 255]
            x, y, w_box, h_box = cv2.boundingRect(points)
            final_crop = cropped_bgr[y:y+h_box, x:x+w_box]
    return final_crop


def unwarp(image: np.ndarray, model: Unwarp) -> np.ndarray:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    unwraped_pil = model.inference(pil_img)
    unwraped_rgb = np.array(unwraped_pil)
    unwraped_bgr = cv2.cvtColor(unwraped_rgb, cv2.COLOR_RGB2BGR)
    return unwraped_bgr


def remove_finger_and_shadow(
    image_bgr: np.ndarray,
    remove_finger: bool = False,
    remove_shadow: bool = False
) -> tuple[np.ndarray, list[str]]:
    """
    指の映り込み除去および影・照明ムラ補正を行う
    """
    result = image_bgr.copy()
    notes = []
    h, w = result.shape[:2]

    # 1. 指の映り込み除去（外周部に接する肌色領域を検出して周囲の紙面色で補完）
    if remove_finger:
        ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        # YCrCb と HSV の複合肌色フィルタ
        mask_ycrcb = cv2.inRange(ycrcb, np.array([40, 133, 77], dtype=np.uint8), np.array([255, 178, 128], dtype=np.uint8))
        mask_hsv = cv2.inRange(hsv, np.array([0, 28, 50], dtype=np.uint8), np.array([25, 180, 255], dtype=np.uint8))
        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # ノイズ除去と膨張
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)

        # ページ外周18%ゾーン（ページ内部の人物イラスト・写真の誤検出を防ぐ）
        margin_x = max(10, int(w * 0.18))
        margin_y = max(10, int(h * 0.18))
        border_zone = np.zeros((h, w), dtype=np.uint8)
        border_zone[:margin_y, :] = 255
        border_zone[h - margin_y:, :] = 255
        border_zone[:, :margin_x] = 255
        border_zone[:, w - margin_x:] = 255

        # 外周の端（2%以内）に実際に接触している連結成分だけを「指」と判定
        edge_touch = np.zeros((h, w), dtype=np.uint8)
        edge_px_x = max(5, int(w * 0.02))
        edge_px_y = max(5, int(h * 0.02))
        edge_touch[:edge_px_y, :] = 255
        edge_touch[h - edge_px_y:, :] = 255
        edge_touch[:, :edge_px_x] = 255
        edge_touch[:, w - edge_px_x:] = 255

        candidate_mask = cv2.bitwise_and(skin_mask, border_zone)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
        finger_mask = np.zeros((h, w), dtype=np.uint8)
        min_area = int(h * w * 0.0005)
        max_area = int(h * w * 0.08)

        finger_detected = False
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                comp = (labels == lbl).astype(np.uint8) * 255
                if np.any(cv2.bitwise_and(comp, edge_touch) > 0):
                    finger_mask = cv2.bitwise_or(finger_mask, comp)
                    finger_detected = True

        if finger_detected:
            # 爪や指先の輪郭も含めるため少し広めに膨張してからインペインティング
            dilate_size = max(11, (min(h, w) // 60) | 1)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            finger_mask = cv2.dilate(finger_mask, kernel_dilate, iterations=1)
            result = cv2.inpaint(result, finger_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            notes.append("指除去")

    # 2. 影・照明ムラ補正（LAB色空間のLチャンネルから背景照度を推定して均一化）
    if remove_shadow:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # 文字や細い線を消すため大きめのモルフォロジー膨張＋ガウシアン平滑化で紙面背景だけを抽出
        k_size = max(31, (min(h, w) // 18) | 1)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bg_l = cv2.dilate(l_chan, kernel_bg)
        bg_l = cv2.GaussianBlur(bg_l, (k_size, k_size), 0)

        # 紙面の基準白レベル（上位パーセンタイル）に合わせて除算正規化
        target_white = float(np.percentile(bg_l, 95))
        target_white = max(210.0, min(252.0, target_white))

        l_float = l_chan.astype(np.float32)
        bg_float = np.maximum(bg_l.astype(np.float32), 1.0)
        corrected_l = np.clip((l_float / bg_float) * target_white, 0, 255).astype(np.uint8)

        # 元画像と適度にブレンドし、過度な白飛びやノイズ増幅を防止
        blended_l = cv2.addWeighted(corrected_l, 0.85, l_chan, 0.15, 0)
        corrected_lab = cv2.merge([blended_l, a_chan, b_chan])
        result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        notes.append("影補正")

    return result, notes


def page_cut(image_bgr: np.ndarray, page_order: str, split_method: str = "smart", offset_pct: float = 0.0) -> list[np.ndarray]:
    """
    見開きページを左右に分割する
    - split_method == "center": 画像中央(50%) + オフセットで分割
    - split_method == "smart": 文字やイラストを避けつつノド(綴じ目)の谷間を検出して分割
    """
    height, width, _ = image_bgr.shape
    offset_px = int(width * (offset_pct / 100.0))

    if split_method == "center":
        split_x = (width // 2) + offset_px
    else:
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        y_start = int(height * 0.10)
        y_end = int(height * 0.90)
        roi_gray = img_gray[y_start:y_end, :]

        search_start = int(width * 0.35)
        search_end = int(width * 0.65)
        roi_center = roi_gray[:, search_start:search_end].astype(np.float32)

        sobel_x = np.abs(cv2.Sobel(roi_center, cv2.CV_32F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(roi_center, cv2.CV_32F, 0, 1, ksize=3))
        edge_profile = np.mean(sobel_x + sobel_y, axis=0)

        brightness_profile = np.mean(roi_center, axis=0)

        kernel_size = max(5, int(width * 0.008) | 1)
        kernel = np.ones(kernel_size) / kernel_size
        edge_smooth = np.convolve(edge_profile, kernel, mode="same")
        bright_smooth = np.convolve(brightness_profile, kernel, mode="same")

        rel_center = (width // 2) - search_start
        x_indices = np.arange(len(edge_smooth))
        dist_penalty = np.abs(x_indices - rel_center) / float(width)

        edge_norm = edge_smooth / (np.max(edge_smooth) + 1e-5)
        bright_norm = bright_smooth / 255.0

        score = (edge_norm * 2.5) + (bright_norm * 0.6) + (dist_penalty * 1.2)

        best_split_idx = int(np.argmin(score))
        split_x = search_start + best_split_idx + offset_px

    split_x = max(10, min(width - 10, split_x))

    left_page = image_bgr[0:height, 0:split_x]
    right_page = image_bgr[0:height, split_x:width]

    if page_order == "left_to_right":
        return [left_page, right_page]
    elif page_order == "right_to_left":
        return [right_page, left_page]
    return [image_bgr]


def detect_and_fix_orientation(image_bgr: np.ndarray, mode: str, page_index: int) -> tuple[np.ndarray, str]:
    if mode == "none":
        return image_bgr, ""
    elif mode == "rotate_90":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), "90°回転"
    elif mode == "rotate_180":
        return cv2.rotate(image_bgr, cv2.ROTATE_180), "180°反転"
    elif mode == "rotate_270":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "270°回転"
    elif mode == "even_180":
        if page_index % 2 == 1:
            return cv2.rotate(image_bgr, cv2.ROTATE_180), "偶数枚目180°反転"
        return image_bgr, ""
    elif mode == "odd_180":
        if page_index % 2 == 0:
            return cv2.rotate(image_bgr, cv2.ROTATE_180), "奇数枚目180°反転"
        return image_bgr, ""
    elif mode == "auto_ocr":
        if not HAS_TESSERACT:
            return image_bgr, "OCR未導入のためスキップ"
        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            osd = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)
            rotate_angle = osd.get("rotate", 0)
            if rotate_angle == 90:
                return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), "OCR自動補正(90°)"
            elif rotate_angle == 180:
                return cv2.rotate(image_bgr, cv2.ROTATE_180), "OCR自動補正(180°)"
            elif rotate_angle == 270:
                return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "OCR自動補正(270°)"
            return image_bgr, ""
        except Exception:
            return image_bgr, "OCR判定不能"
    return image_bgr, ""


def img_format(
    image_path: str,
    croper: YOLO,
    unwarper: Unwarp,
    page_order: str,
    device: str | int,
    orient_mode: str,
    img_index: int,
    split_method: str,
    offset_pct: float,
    remove_finger: bool = False,
    remove_shadow: bool = False
) -> tuple[list[np.ndarray], list[str]]:
    notes = []
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("画像の読み込みに失敗しました（非対応フォーマットまたは破損ファイル）")

    croped_image = crop(image, croper, device)
    unwarped_image = unwarp(croped_image, unwarper)

    # 指除去・影補正（見開き分割の前に実施し、ノド中央の影もまとめて補正）
    if remove_finger or remove_shadow:
        unwarped_image, fs_notes = remove_finger_and_shadow(
            unwarped_image,
            remove_finger=remove_finger,
            remove_shadow=remove_shadow
        )
        notes.extend(fs_notes)

    if page_order in ["left_to_right", "right_to_left"]:
        cutted_images = page_cut(unwarped_image, page_order, split_method, offset_pct)
    else:
        cutted_images = [unwarped_image]

    final_pages = []
    for sub_idx, page_img in enumerate(cutted_images):
        current_idx = img_index * len(cutted_images) + sub_idx
        fixed_img, note = detect_and_fix_orientation(page_img, orient_mode, current_idx)
        if note:
            notes.append(note)
        final_pages.append(fixed_img)

    return final_pages, notes


# ==========================================
# アーカイブ / 電子書籍ビルダー
# ==========================================
def build_zip(png_bytes_list: list[bytes], output_path: str, compress: bool = True, progress_cb=None):
    """連番画像をまとめた ZIP または CBZ ファイルを生成"""
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    total = len(png_bytes_list)
    with zipfile.ZipFile(output_path, "w", compression=compression) as zf:
        for idx, img_bytes in enumerate(png_bytes_list):
            zf.writestr(f"{idx:04d}.png", img_bytes)
            if progress_cb and (idx % 5 == 0 or idx == total - 1):
                progress_cb(idx + 1, total)


def build_cbr(png_bytes_list: list[bytes], output_path: str, rar_exe: str, progress_cb=None):
    """連番画像をまとめた CBR (RARアーカイブ) ファイルを生成"""
    total = len(png_bytes_list)
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, img_bytes in enumerate(png_bytes_list):
            img_path = os.path.join(tmpdir, f"{idx:04d}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            if progress_cb and (idx % 5 == 0 or idx == total - 1):
                progress_cb((idx + 1) * 0.7, total)

        if os.path.exists(output_path):
            os.remove(output_path)

        cmd = [rar_exe, "a", "-ep", "-idq", os.path.abspath(output_path), "*.png"]
        proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError(f"CBR作成エラー: {proc.stderr or proc.stdout}")
        if progress_cb:
            progress_cb(total, total)


def build_epub(
    pages_meta: list[tuple[bytes, int, int]],
    output_path: str,
    title: str = "BookScan Output",
    direction: str = "rtl",
    progress_cb=None
):
    """
    固定レイアウト (Fixed-Layout) EPUB 3.0 ファイルを生成
    pages_meta: [(png_bytes, width, height), ...]
    direction: "rtl" (右開き・縦書き) または "ltr" (左開き・横書き)
    """
    book_id = f"urn:uuid:{uuid.uuid4()}"
    modified_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total = len(pages_meta)

    with zipfile.ZipFile(output_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
        zf.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)

        manifest_items = [
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            '<item id="css" href="style.css" media-type="text/css"/>',
        ]
        spine_items = []
        nav_items = []

        for idx, (img_bytes, w, h) in enumerate(pages_meta):
            page_num = idx + 1
            img_name = f"images/page_{idx:04d}.png"
            xhtml_name = f"text/page_{idx:04d}.xhtml"

            zf.writestr(f"OEBPS/{img_name}", img_bytes, compress_type=zipfile.ZIP_STORED)

            cover_prop = ' properties="cover-image"' if idx == 0 else ""
            manifest_items.append(
                f'<item id="img_{idx:04d}" href="{img_name}" media-type="image/png"{cover_prop}/>'
            )
            manifest_items.append(
                f'<item id="page_{idx:04d}" href="{xhtml_name}" media-type="application/xhtml+xml" properties="svg"/>'
            )

            spread_side = ""
            if direction == "rtl":
                spread_side = ' properties="page-spread-right"' if idx % 2 == 0 else ' properties="page-spread-left"'
            else:
                spread_side = ' properties="page-spread-left"' if idx % 2 == 0 else ' properties="page-spread-right"'

            spine_items.append(f'<itemref idref="page_{idx:04d}"{spread_side}/>')
            nav_items.append(f'<li><a href="{xhtml_name}">Page {page_num}</a></li>')

            page_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <meta charset="UTF-8"/>
  <title>Page {page_num}</title>
  <meta name="viewport" content="width={w}, height={h}"/>
  <link rel="stylesheet" type="text/css" href="../style.css"/>
</head>
<body>
  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
       version="1.1" width="100%" height="100%" viewBox="0 0 {w} {h}">
    <image width="{w}" height="{h}" xlink:href="../{img_name}"/>
  </svg>
</body>
</html>"""
            zf.writestr(f"OEBPS/{xhtml_name}", page_xhtml, compress_type=zipfile.ZIP_DEFLATED)

            if progress_cb and (idx % 5 == 0 or idx == total - 1):
                progress_cb(idx + 1, total)

        style_css = """@charset "UTF-8";
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background-color: #000000;
}
svg {
  display: block;
  width: 100%;
  height: 100%;
}"""
        zf.writestr("OEBPS/style.css", style_css, compress_type=zipfile.ZIP_DEFLATED)

        nav_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <meta charset="UTF-8"/>
  <title>目次</title>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>目次</h1>
    <ol>
      {chr(10).join(nav_items)}
    </ol>
  </nav>
</body>
</html>"""
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml, compress_type=zipfile.ZIP_DEFLATED)

        safe_title = (
            title.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        content_opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId"
         prefix="rendition: http://www.idpf.org/vocab/rendition/#">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">{book_id}</dc:identifier>
    <dc:title>{safe_title}</dc:title>
    <dc:language>ja</dc:language>
    <meta property="dcterms:modified">{modified_iso}</meta>
    <meta property="rendition:layout">pre-paginated</meta>
    <meta property="rendition:orientation">auto</meta>
    <meta property="rendition:spread">auto</meta>
  </metadata>
  <manifest>
    {chr(10).join(manifest_items)}
  </manifest>
  <spine page-progression-direction="{direction}">
    {chr(10).join(spine_items)}
  </spine>
</package>"""
        zf.writestr("OEBPS/content.opf", content_opf, compress_type=zipfile.ZIP_DEFLATED)


# ==========================================
# CustomTkinter GUIアプリケーション本体
# ==========================================
class BookScannerModernGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("BookScan AI - 書籍・書類スキャン補正ツール")
        self.geometry("800x920")
        self.minsize(750, 880)

        if os.path.exists(".\\book.ico"):
            self.iconbitmap(default=".\\book.ico")

        self.is_running = False
        self.stop_requested = False

        # 変数定義
        self.input_dir_var = ctk.StringVar()
        self.output_dir_var = ctk.StringVar()

        # 出力形式フラグ
        self.is_pdf_var = ctk.BooleanVar(value=True)
        self.is_epub_var = ctk.BooleanVar(value=False)
        self.is_cbz_var = ctk.BooleanVar(value=False)
        self.is_cbr_var = ctk.BooleanVar(value=False)
        self.is_zip_var = ctk.BooleanVar(value=False)
        self.is_img_var = ctk.BooleanVar(value=False)

        self.base_name_var = ctk.StringVar(value="output")
        self.epub_dir_var = ctk.StringVar(value="rtl")

        self.use_gpu_var = ctk.BooleanVar(value=torch.cuda.is_available())
        self.page_order_var = ctk.StringVar(value="not_cut")
        self.split_method_var = ctk.StringVar(value="smart")
        self.split_offset_var = ctk.DoubleVar(value=0.0)
        self.remove_finger_var = ctk.BooleanVar(value=False)
        self.remove_shadow_var = ctk.BooleanVar(value=False)

        self.orient_map = {
            "補正しない": "none",
            "自動判定 (OCRで上下左右を自動修正)": "auto_ocr",
            "偶数枚目だけ 180° 反転 (上下交互スキャン用)": "even_180",
            "奇数枚目だけ 180° 反転 (上下交互スキャン用)": "odd_180",
            "すべて 180° 反転 (逆さま一括修正)": "rotate_180",
            "すべて 時計回りに 90° 回転": "rotate_90",
            "すべて 反時計回りに 90° 回転": "rotate_270",
        }
        self.orient_display_var = ctk.StringVar(value="補正しない")

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # --- ヘッダー ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 4))
        header_frame.grid_columnconfigure(0, weight=1)

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="📚 BookScan AI",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_lbl.grid(row=0, column=0, sticky="w")

        sub_lbl = ctk.CTkLabel(
            header_frame,
            text="YOLO自動切り抜き ・ UVDoc歪み補正 ・ 指/影除去 ・ 見開き分割 ・ PDF / EPUB / CBZ / CBR / ZIP出力",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        sub_lbl.grid(row=1, column=0, sticky="w")

        theme_menu = ctk.CTkOptionMenu(
            header_frame,
            values=["System", "Dark", "Light"],
            command=ctk.set_appearance_mode,
            width=100,
            height=28
        )
        theme_menu.set("System")
        theme_menu.grid(row=0, column=1, rowspan=2, sticky="e")

        # --- 1. フォルダ設定カード ---
        dir_card = ctk.CTkFrame(self, corner_radius=12)
        dir_card.grid(row=1, column=0, sticky="ew", padx=20, pady=6)
        dir_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(dir_card, text="1. 入出力フォルダ", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 4)
        )

        ctk.CTkLabel(dir_card, text="入力フォルダ:").grid(row=1, column=0, sticky="w", padx=(15, 8), pady=5)
        ctk.CTkEntry(dir_card, textvariable=self.input_dir_var, placeholder_text="スキャン画像が入ったフォルダを選択...").grid(
            row=1, column=1, sticky="ew", pady=5
        )
        ctk.CTkButton(dir_card, text="参照", width=70, command=self._select_input_dir).grid(
            row=1, column=2, padx=(8, 15), pady=5
        )

        ctk.CTkLabel(dir_card, text="出力フォルダ:").grid(row=2, column=0, sticky="w", padx=(15, 8), pady=(5, 12))
        ctk.CTkEntry(dir_card, textvariable=self.output_dir_var, placeholder_text="保存先フォルダを選択...").grid(
            row=2, column=1, sticky="ew", pady=(5, 12)
        )
        ctk.CTkButton(dir_card, text="参照", width=70, command=self._select_output_dir).grid(
            row=2, column=2, padx=(8, 15), pady=(5, 12)
        )

        # --- 2. 処理オプションカード ---
        opt_card = ctk.CTkFrame(self, corner_radius=12)
        opt_card.grid(row=2, column=0, sticky="ew", padx=20, pady=6)
        opt_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opt_card, text="2. 出力・補正オプション", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 4)
        )

        # 出力形式（複数選択可）
        ctk.CTkLabel(opt_card, text="出力形式:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        fmt_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        fmt_row.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=5)

        ctk.CTkCheckBox(fmt_row, text="PDF", variable=self.is_pdf_var, width=62).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(fmt_row, text="EPUB", variable=self.is_epub_var, width=68, command=self._on_epub_toggle).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(fmt_row, text="CBZ", variable=self.is_cbz_var, width=62).pack(side="left", padx=(0, 8))

        cbr_state = "normal" if RAR_EXE else "disabled"
        self.cbr_chk = ctk.CTkCheckBox(fmt_row, text="CBR", variable=self.is_cbr_var, width=62, state=cbr_state)
        self.cbr_chk.pack(side="left", padx=(0, 8))

        ctk.CTkCheckBox(fmt_row, text="ZIP", variable=self.is_zip_var, width=60).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(fmt_row, text="画像(PNG)", variable=self.is_img_var, width=90).pack(side="left")

        if not RAR_EXE:
            ctk.CTkLabel(
                fmt_row,
                text="(CBR: WinRAR未検出)",
                text_color="#e67e22",
                font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=(4, 0))

        # 出力ファイル名 & EPUB綴じ方向
        ctk.CTkLabel(opt_card, text="出力ファイル名:").grid(row=2, column=0, sticky="w", padx=(15, 10), pady=5)
        name_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        name_row.grid(row=2, column=1, sticky="ew", padx=(0, 15), pady=5)

        ctk.CTkEntry(name_row, textvariable=self.base_name_var, width=180, placeholder_text="output").pack(side="left")
        ctk.CTkLabel(
            name_row,
            text="※拡張子は自動付与",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(6, 16))

        ctk.CTkLabel(name_row, text="EPUB綴じ方向:").pack(side="left", padx=(0, 6))
        self.epub_dir_seg = ctk.CTkSegmentedButton(
            name_row,
            values=["右開き (縦書き)", "左開き (横書き)"],
            command=self._on_epub_dir_change,
            width=210,
            state=ctk.DISABLED
        )
        self.epub_dir_seg.set("右開き (縦書き)")
        self.epub_dir_seg.pack(side="left")

        # 見開き分割（ページ順）
        ctk.CTkLabel(opt_card, text="見開き分割:").grid(row=3, column=0, sticky="w", padx=(15, 10), pady=5)
        self.seg_btn = ctk.CTkSegmentedButton(
            opt_card,
            values=["分割しない", "左 → 右 (横書き)", "右 → 左 (縦書き)"],
            command=self._on_page_order_change
        )
        self.seg_btn.set("分割しない")
        self.seg_btn.grid(row=3, column=1, sticky="w", padx=(0, 15), pady=5)

        # 分割アルゴリズム＆微調整
        ctk.CTkLabel(opt_card, text="分割位置:").grid(row=4, column=0, sticky="w", padx=(15, 10), pady=5)
        split_detail_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        split_detail_row.grid(row=4, column=1, sticky="ew", padx=(0, 15), pady=5)

        self.split_method_seg = ctk.CTkSegmentedButton(
            split_detail_row,
            values=["自動検出 (文字回避)", "中央 50% 固定"],
            command=self._on_split_method_change,
            width=230,
            state=ctk.DISABLED
        )
        self.split_method_seg.set("自動検出 (文字回避)")
        self.split_method_seg.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(split_detail_row, text="微調整:").pack(side="left")
        self.offset_slider = ctk.CTkSlider(
            split_detail_row,
            from_=-10.0,
            to=10.0,
            number_of_steps=40,
            variable=self.split_offset_var,
            command=self._on_offset_change,
            width=120,
            state=ctk.DISABLED
        )
        self.offset_slider.pack(side="left", padx=6)
        self.offset_lbl = ctk.CTkLabel(split_detail_row, text="0.0%", width=48)
        self.offset_lbl.pack(side="left")

        # 向き補正
        ctk.CTkLabel(opt_card, text="向き補正:").grid(row=5, column=0, sticky="w", padx=(15, 10), pady=5)
        orient_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        orient_row.grid(row=5, column=1, sticky="ew", padx=(0, 15), pady=5)

        ctk.CTkOptionMenu(
            orient_row,
            values=list(self.orient_map.keys()),
            variable=self.orient_display_var,
            width=310
        ).pack(side="left")

        ocr_badge = "OCR利用可" if HAS_TESSERACT else "OCR未導入"
        ocr_color = "#2ecc71" if HAS_TESSERACT else "#e67e22"
        ctk.CTkLabel(orient_row, text=f"[{ocr_badge}]", text_color=ocr_color, font=ctk.CTkFont(size=11)).pack(
            side="left", padx=10
        )

        # 指・影補正（ON/OFFスイッチ）
        ctk.CTkLabel(opt_card, text="指・影補正:").grid(row=6, column=0, sticky="w", padx=(15, 10), pady=5)
        clean_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        clean_row.grid(row=6, column=1, sticky="w", padx=(0, 15), pady=5)

        ctk.CTkSwitch(
            clean_row,
            text="指の映り込み除去",
            variable=self.remove_finger_var
        ).pack(side="left", padx=(0, 20))

        ctk.CTkSwitch(
            clean_row,
            text="影・照明ムラ補正",
            variable=self.remove_shadow_var
        ).pack(side="left")

        # GPU設定
        ctk.CTkLabel(opt_card, text="高速化(GPU):").grid(row=7, column=0, sticky="w", padx=(15, 10), pady=(5, 12))
        gpu_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        gpu_row.grid(row=7, column=1, sticky="w", padx=(0, 15), pady=(5, 12))

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_text = f"GPU (CUDA) を有効にする  —  検出済み: {gpu_name}"
        else:
            gpu_text = "GPU (CUDA) を有効にする  —  未検出 (CPUで動作)"

        ctk.CTkSwitch(gpu_row, text=gpu_text, variable=self.use_gpu_var).pack(side="left")

        # --- 3. 実行・2段階進捗カード ---
        prog_card = ctk.CTkFrame(self, corner_radius=12)
        prog_card.grid(row=3, column=0, sticky="ew", padx=20, pady=6)
        prog_card.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(prog_card, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 8))

        self.start_btn = ctk.CTkButton(
            btn_row,
            text="▶ 処理を開始する",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=38,
            width=180,
            command=self._start_process
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            btn_row,
            text="■ 中断",
            fg_color="#c0392b",
            hover_color="#962d22",
            height=38,
            width=90,
            state="disabled",
            command=self._stop_process
        )
        self.stop_btn.pack(side="left")

        self.overall_status_lbl = ctk.CTkLabel(
            btn_row,
            text="待機中",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.overall_status_lbl.pack(side="right")

        # ステップ1：画像補正進捗バー
        step1_lbl_row = ctk.CTkFrame(prog_card, fg_color="transparent")
        step1_lbl_row.grid(row=1, column=0, sticky="ew", padx=15, pady=(2, 2))
        ctk.CTkLabel(step1_lbl_row, text="ステップ 1：画像補正・分割・向き修正", font=ctk.CTkFont(size=12)).pack(side="left")
        self.img_prog_text = ctk.CTkLabel(step1_lbl_row, text="0 / 0 枚 (0%)", font=ctk.CTkFont(size=12))
        self.img_prog_text.pack(side="right")

        self.img_progress_bar = ctk.CTkProgressBar(prog_card, height=12)
        self.img_progress_bar.set(0)
        self.img_progress_bar.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 8))

        # ステップ2：1ファイル結合・書き出し進捗バー（PDF / EPUB / CBZ / CBR / ZIP 共通）
        step2_lbl_row = ctk.CTkFrame(prog_card, fg_color="transparent")
        step2_lbl_row.grid(row=3, column=0, sticky="ew", padx=15, pady=(2, 2))
        ctk.CTkLabel(step2_lbl_row, text="ステップ 2：1ファイル結合・書き出し", font=ctk.CTkFont(size=12)).pack(side="left")
        self.bundle_prog_text = ctk.CTkLabel(step2_lbl_row, text="待機中 (0 ページ)", font=ctk.CTkFont(size=12))
        self.bundle_prog_text.pack(side="right")

        self.bundle_progress_bar = ctk.CTkProgressBar(prog_card, height=12, progress_color="#2ecc71")
        self.bundle_progress_bar.set(0)
        self.bundle_progress_bar.grid(row=4, column=0, sticky="ew", padx=15, pady=(0, 12))

        # --- 4. 実行ログカード ---
        log_card = ctk.CTkFrame(self, corner_radius=12)
        log_card.grid(row=4, column=0, sticky="nsew", padx=20, pady=(6, 14))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_card, text="実行ログ", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=15, pady=(8, 2)
        )

        self.log_box = ctk.CTkTextbox(log_card, font=ctk.CTkFont(family="Consolas", size=12), state="disabled")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 12))

    # --- UIイベントハンドラ ---
    def _on_epub_toggle(self):
        if self.is_epub_var.get():
            self.epub_dir_seg.configure(state=ctk.NORMAL)
        else:
            self.epub_dir_seg.configure(state=ctk.DISABLED)

    def _on_epub_dir_change(self, value: str):
        if value == "左開き (横書き)":
            self.epub_dir_var.set("ltr")
        else:
            self.epub_dir_var.set("rtl")

    def _on_page_order_change(self, value: str):
        mapping = {
            "分割しない": "not_cut",
            "左 → 右 (横書き)": "left_to_right",
            "右 → 左 (縦書き)": "right_to_left"
        }
        self.page_order_var.set(mapped := mapping.get(value, "not_cut"))
        if mapped == "not_cut":
            self.split_method_seg.configure(state=ctk.DISABLED)
            self.offset_slider.configure(state=ctk.DISABLED)
        else:
            self.split_method_seg.configure(state=ctk.NORMAL)
            self.offset_slider.configure(state=ctk.NORMAL)
            if mapped == "left_to_right":
                self.epub_dir_var.set("ltr")
                self.epub_dir_seg.set("左開き (横書き)")
            elif mapped == "right_to_left":
                self.epub_dir_var.set("rtl")
                self.epub_dir_seg.set("右開き (縦書き)")

    def _on_split_method_change(self, value: str):
        if value == "中央 50% 固定":
            self.split_method_var.set("center")
        else:
            self.split_method_var.set("smart")

    def _on_offset_change(self, value: float):
        self.offset_lbl.configure(text=f"{value:+.1f}%")

    def _select_input_dir(self):
        path = fd.askdirectory(title="入力ディレクトリ選択")
        if path:
            self.input_dir_var.set(path)

    def _select_output_dir(self):
        path = fd.askdirectory(title="出力ディレクトリ選択")
        if path:
            self.output_dir_var.set(path)

    def _log(self, message: str):
        def append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, append)

    def _update_img_progress(self, current: int, total: int, status_text: str):
        def update():
            ratio = (current / total) if total > 0 else 0.0
            pct = int(ratio * 100)
            self.img_progress_bar.set(ratio)
            self.img_prog_text.configure(text=f"{current} / {total} 枚 ({pct}%)")
            self.overall_status_lbl.configure(text=status_text)
        self.after(0, update)

    def _update_bundle_progress(self, ratio: float, detail_text: str, status_text: str = None):
        def update():
            clamped = max(0.0, min(1.0, ratio))
            self.bundle_progress_bar.set(clamped)
            self.bundle_prog_text.configure(text=detail_text)
            if status_text:
                self.overall_status_lbl.configure(text=status_text)
        self.after(0, update)

    def _set_ui_running(self, running: bool):
        def toggle():
            self.is_running = running
            self.start_btn.configure(state="disabled" if running else "normal")
            self.stop_btn.configure(state="normal" if running else "disabled")
        self.after(0, toggle)

    def _stop_process(self):
        if self.is_running:
            self.stop_requested = True
            self._log("⚠️ 中断リクエストを受け付けました。現在の画像の処理後に停止します...")

    def _start_process(self):
        in_dir = self.input_dir_var.get().strip()
        out_dir = self.output_dir_var.get().strip()
        any_selected = any([
            self.is_pdf_var.get(),
            self.is_epub_var.get(),
            self.is_cbz_var.get(),
            self.is_cbr_var.get(),
            self.is_zip_var.get(),
            self.is_img_var.get(),
        ])

        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showwarning("入力エラー", "有効な入力フォルダを選択してください。")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showwarning("入力エラー", "有効な出力フォルダを選択してください。")
            return
        if not any_selected:
            messagebox.showwarning("設定エラー", "出力形式を少なくとも1つ選択してください。")
            return

        self.stop_requested = False
        self.img_progress_bar.set(0)
        self.bundle_progress_bar.set(0)
        self.img_prog_text.configure(text="0 / 0 枚 (0%)")
        self.bundle_prog_text.configure(text="待機中")
        self._set_ui_running(True)

        thread = threading.Thread(target=self._run_worker, daemon=True)
        thread.start()

    def _run_worker(self):
        in_dir = self.input_dir_var.get().strip()
        out_dir = self.output_dir_var.get().strip()

        is_pdf = self.is_pdf_var.get()
        is_epub = self.is_epub_var.get()
        is_cbz = self.is_cbz_var.get()
        is_cbr = self.is_cbr_var.get() and (RAR_EXE is not None)
        is_zip = self.is_zip_var.get()
        is_img = self.is_img_var.get()
        need_bundle = any([is_pdf, is_epub, is_cbz, is_cbr, is_zip])

        page_order = self.page_order_var.get()
        split_method = self.split_method_var.get()
        offset_pct = self.split_offset_var.get()
        use_gpu = self.use_gpu_var.get()
        orient_label = self.orient_display_var.get()
        orient_mode = self.orient_map.get(orient_label, "none")
        remove_finger = self.remove_finger_var.get()
        remove_shadow = self.remove_shadow_var.get()
        epub_direction = self.epub_dir_var.get()

        raw_name = self.base_name_var.get().strip() or "output"
        base_name = os.path.splitext(raw_name)[0] if os.path.splitext(raw_name)[1].lower() in {
            ".pdf", ".epub", ".cbz", ".cbr", ".zip"
        } else raw_name

        try:
            if use_gpu and torch.cuda.is_available():
                yolo_device = 0
                unwarper = Unwarp(providers=["CUDAExecutionProvider"])
                self._log(f"🚀 GPUモード起動: {torch.cuda.get_device_name(0)}")
            else:
                if use_gpu:
                    self._log("⚠️ CUDA環境が未検出のため、CPUモードに切り替えます。")
                yolo_device = "cpu"
                unwarper = Unwarp()
                self._log("💻 CPUモード起動")

            self._log("📦 YOLOモデル (yolo26s-seg.pt) を読み込んでいます...")
            croper = YOLO("yolo26s-seg.pt").to(yolo_device)

            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
            img_name_list = sorted(
                [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in valid_exts],
                key=natural_sort_key
            )

            total_imgs = len(img_name_list)
            if total_imgs == 0:
                self._log("❌ 入力フォルダ内に画像ファイルが見つかりませんでした。")
                self._update_img_progress(0, 0, "画像なし")
                return

            selected_fmts = [
                name for name, flag in [
                    ("PDF", is_pdf), ("EPUB", is_epub), ("CBZ", is_cbz),
                    ("CBR", is_cbr), ("ZIP", is_zip), ("PNG", is_img)
                ] if flag
            ]
            clean_opts = []
            if remove_finger:
                clean_opts.append("指除去ON")
            if remove_shadow:
                clean_opts.append("影補正ON")
            clean_str = ", ".join(clean_opts) if clean_opts else "指・影補正OFF"

            self._log(
                f"📂 対象: {total_imgs} 枚 | 出力: {', '.join(selected_fmts)} | "
                f"分割: {page_order} | 向き: {orient_label} | {clean_str}"
            )

            png_bytes_list: list[bytes] = []
            pages_meta: list[tuple[bytes, int, int]] = []
            page_counter = 0

            # --- ステップ1：画像補正ループ ---
            for i, img_name in enumerate(img_name_list):
                if self.stop_requested:
                    self._log("🛑 処理を中断しました。")
                    self._update_img_progress(i, total_imgs, "中断完了")
                    return

                self._update_img_progress(i, total_imgs, f"[1/2] 画像処理中 ({i+1}/{total_imgs})")
                img_path = os.path.join(in_dir, img_name)

                try:
                    img_list, notes = img_format(
                        img_path, croper, unwarper, page_order, yolo_device,
                        orient_mode, i, split_method, offset_pct,
                        remove_finger=remove_finger,
                        remove_shadow=remove_shadow
                    )

                    for page_img in img_list:
                        if is_img:
                            img_output_path = os.path.join(out_dir, f"{page_counter:04d}.png")
                            cv2.imwrite(img_output_path, page_img)

                        if need_bundle:
                            success, encoded_img = cv2.imencode(".png", page_img)
                            if success:
                                img_bytes = encoded_img.tobytes()
                                png_bytes_list.append(img_bytes)
                                if is_epub:
                                    ph, pw = page_img.shape[:2]
                                    pages_meta.append((img_bytes, pw, ph))

                        page_counter += 1

                    note_str = f" [{', '.join(notes)}]" if notes else ""
                    self._log(f"✓ {img_name} -> {len(img_list)}ページ出力{note_str}")

                except Exception as e:
                    self._log(f"✗ エラー ({img_name}): {e}")

            self._update_img_progress(total_imgs, total_imgs, "[1/2] 画像処理完了")

            # --- ステップ2：1ファイル結合・書き出し（PDF / EPUB / CBZ / CBR / ZIP 共通進捗バー） ---
            bundle_tasks = [
                fmt for fmt, flag in [
                    ("PDF", is_pdf), ("EPUB", is_epub), ("CBZ", is_cbz),
                    ("CBR", is_cbr), ("ZIP", is_zip)
                ] if flag
            ]

            if bundle_tasks and png_bytes_list:
                total_pages = len(png_bytes_list)
                num_tasks = len(bundle_tasks)

                for t_idx, fmt in enumerate(bundle_tasks):
                    if self.stop_requested:
                        self._log("🛑 ファイル結合を中断しました。")
                        return

                    base_ratio = t_idx / num_tasks
                    span_ratio = 1.0 / num_tasks

                    def report_sub_progress(sub_ratio: float, msg: str):
                        overall = base_ratio + (span_ratio * sub_ratio)
                        pct = int(overall * 100)
                        self._update_bundle_progress(
                            overall,
                            f"{fmt} 作成中 ({t_idx+1}/{num_tasks}): {msg} ({pct}%)",
                            f"[2/2] {fmt} 出力中..."
                        )

                    if fmt == "PDF":
                        pdf_filename = f"{base_name}.pdf"
                        pdf_output_path = os.path.join(out_dir, pdf_filename)
                        self._log(f"📄 全 {total_pages} ページをPDFに結合しています...")
                        report_sub_progress(0.10, f"変換準備中 ({total_pages} ページ)")

                        done_event = threading.Event()
                        pdf_result = {}

                        def convert_worker():
                            try:
                                pdf_result["bytes"] = img2pdf.convert(png_bytes_list)
                            except Exception as exc:
                                pdf_result["error"] = exc
                            finally:
                                done_event.set()

                        conv_thread = threading.Thread(target=convert_worker, daemon=True)
                        conv_thread.start()

                        start_time = time.time()
                        sim_sub = 0.15
                        while not done_event.is_set():
                            if self.stop_requested:
                                self._log("🛑 PDF生成の待機を中断しました。")
                                return
                            elapsed = time.time() - start_time
                            sim_sub = min(0.85, sim_sub + (0.85 - sim_sub) * 0.08)
                            report_sub_progress(sim_sub, f"結合処理中 ({total_pages} ページ / {elapsed:.1f}秒)")
                            time.sleep(0.15)

                        if "error" in pdf_result:
                            raise pdf_result["error"]

                        report_sub_progress(0.92, f"書き出し中 ({pdf_filename})")
                        with open(pdf_output_path, "wb") as f_out:
                            f_out.write(pdf_result["bytes"])
                        report_sub_progress(1.0, f"{pdf_filename} 完了")
                        self._log(f"✨ PDFの作成が完了しました ({total_pages} ページ): {pdf_output_path}")

                    elif fmt == "EPUB":
                        epub_filename = f"{base_name}.epub"
                        epub_output_path = os.path.join(out_dir, epub_filename)
                        dir_label = "右開き" if epub_direction == "rtl" else "左開き"
                        self._log(f"📖 全 {total_pages} ページをEPUB ({dir_label}) に結合しています...")
                        report_sub_progress(0.05, "パッケージ構築開始")

                        build_epub(
                            pages_meta,
                            epub_output_path,
                            title=base_name,
                            direction=epub_direction,
                            progress_cb=lambda cur, tot: report_sub_progress(cur / tot, f"{cur}/{tot} ページ")
                        )
                        self._log(f"✨ EPUBの作成が完了しました ({total_pages} ページ): {epub_output_path}")

                    elif fmt == "CBZ":
                        cbz_filename = f"{base_name}.cbz"
                        cbz_output_path = os.path.join(out_dir, cbz_filename)
                        self._log(f"🗜️ 全 {total_pages} ページをCBZに結合しています...")
                        report_sub_progress(0.05, "アーカイブ作成開始")

                        build_zip(
                            png_bytes_list,
                            cbz_output_path,
                            compress=False,
                            progress_cb=lambda cur, tot: report_sub_progress(cur / tot, f"{cur}/{tot} ページ")
                        )
                        self._log(f"✨ CBZの作成が完了しました ({total_pages} ページ): {cbz_output_path}")

                    elif fmt == "CBR":
                        cbr_filename = f"{base_name}.cbr"
                        cbr_output_path = os.path.join(out_dir, cbr_filename)
                        self._log(f"📦 全 {total_pages} ページをCBRに結合しています...")
                        report_sub_progress(0.05, "RAR圧縮準備中")

                        build_cbr(
                            png_bytes_list,
                            cbr_output_path,
                            RAR_EXE,
                            progress_cb=lambda cur, tot: report_sub_progress(cur / tot, f"{int(cur)}/{tot} ページ")
                        )
                        self._log(f"✨ CBRの作成が完了しました ({total_pages} ページ): {cbr_output_path}")

                    elif fmt == "ZIP":
                        zip_filename = f"{base_name}.zip"
                        zip_output_path = os.path.join(out_dir, zip_filename)
                        self._log(f"🗜️ 全 {total_pages} ページをZIPに結合しています...")
                        report_sub_progress(0.05, "ZIP圧縮開始")

                        build_zip(
                            png_bytes_list,
                            zip_output_path,
                            compress=True,
                            progress_cb=lambda cur, tot: report_sub_progress(cur / tot, f"{cur}/{tot} ページ")
                        )
                        self._log(f"✨ ZIPの作成が完了しました ({total_pages} ページ): {zip_output_path}")

                self._update_bundle_progress(
                    1.0,
                    f"完了: {', '.join(bundle_tasks)} ({total_pages} ページ / 100%)",
                    "すべて完了"
                )
            else:
                self._update_bundle_progress(1.0, "スキップ (1ファイル出力オフ)", "すべて完了")

            self.after(0, lambda: messagebox.showinfo("完了", f"すべての処理が完了しました！\n（合計 {page_counter} ページ）"))

        except Exception as e:
            self._log(f"🔥 致命的なエラー: {e}")
            self._update_img_progress(0, 1, "エラー終了")
            self.after(0, lambda: messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{e}"))

        finally:
            self._set_ui_running(False)


if __name__ == "__main__":
    app = BookScannerModernGUI()
    app.mainloop()
