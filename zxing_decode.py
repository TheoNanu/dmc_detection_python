import time

import cv2, zxingcpp
import numpy as np

# for i in range(1, 11):
#     filename = f"synthetic_image{i}.png"
#     img = cv2.imread(f"./test_images/{filename}")
#
#     barcodes = zxingcpp.read_barcodes(img)
#
#     print(i)
#     print(barcodes)
#
#     for barcode in barcodes:
#         print(barcode.text)
#         pos = barcode.position
#         bl = pos.bottom_left
#         br = pos.bottom_right
#         tl = pos.top_left
#         tr = pos.top_right
#
#         cx = (tl.x + br.x) // 2
#         cy = (tl.y + br.y) // 2
#
#         pts = np.array([[bl.x, bl.y], [br.x, br.y], [tr.x, tr.y], [tl.x, tl.y]], dtype=np.int32)
#         cv2.polylines(img, [pts], True, (0, 255, 0), 2)
#
#         # cv2.rectangle(img, (tl.x, tl.y), (br.x, br.y), (0, 255, 0), 2)
#
#         text_size = cv2.getTextSize(barcode.text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         cv2.putText(img, barcode.text,
#                     (int(cx - (text_size[0][0] / 2)), int(cy - (text_size[0][1] / 2))),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 173), 2)
#
#     cv2.imwrite(f"{filename.split(".")[0]}_zxing_result.png", img)

img = cv2.imread(f"./test_images/dmc_on_object_test_image.png")
start = time.time()
barcodes = zxingcpp.read_barcodes(img)
print(f"Execution time: {time.time() - start}")

for barcode in barcodes:
    print(barcode.text)