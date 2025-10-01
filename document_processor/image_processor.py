from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(
    use_textline_orientation=True,
)


def ocr_pdf(path, output_dir="default"):
    if output_dir == "default":
        output_dir = "result/" + path.split("\\")[-1].split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(path)
    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i}.png")
        page.save(img_path, "PNG")
        result = ocr.predict(img_path)
        for line in result:
            line.save_to_img(output_dir)
