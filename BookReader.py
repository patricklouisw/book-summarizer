import pytesseract
from pdf2image import convert_from_path

print("CONVERT_FROM_PATH")
pages = convert_from_path("Infrastructure.pdf")

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract"

with open("Infrastructure-summary.txt", "w+") as f:

    for i in range(len(pages)):
        page_num = "page_{0}.png".format(str(i))
        print(page_num)
        pages[i].save(page_num, "PNG")

        # path_to_png = "C:\\Users\\Patrick\ Louis\\Desktop\\book-summarizer\\{0}".format(
        #     page_num)
        print(pytesseract.image_to_string(page_num))
        f.write(pytesseract.image_to_string(page_num))
