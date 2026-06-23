import time

from pylibdmtx import pylibdmtx
import cv2

# barcodes = []
#
# for i in range(1, 11):
#     filename = f"synthetic_image{i}.png"
#     img = cv2.imread(f"./test_images/{filename}")
#
#     barcodes = pylibdmtx.decode(img, timeout=3000)
#
#     h_img = img.shape[0]
#
#     for barcode in barcodes:
#         r = barcode.rect
#         text = barcode.data.decode("utf-8")
#
#         x1 = r.left
#         y1 = h_img - (r.top + r.height)
#         x2 = x1 + r.width
#         y2 = h_img - r.top
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         cv2.putText(img, text,
#                    (int(cx - (text_size[0][0] / 2)), int(cy - (text_size[0][1] / 2))),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 173), 2)
#
#     cv2.imwrite(f"{filename.split(".")[0]}_pylibdmtx_result.png", img)
img = cv2.imread(f"test_images/dmc_on_object_test_image.png")

start = time.time()
barcode = pylibdmtx.decode(img, timeout=3000)
print(f"Execution time: {time.time() - start}")

print(barcode)