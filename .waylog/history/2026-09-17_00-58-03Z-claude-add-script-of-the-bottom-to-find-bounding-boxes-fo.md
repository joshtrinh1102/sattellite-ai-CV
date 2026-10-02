# add script of the bottom to find bounding boxes fo
_Exported on 10/02/2026 at 13:19:52 GMT-4 from Claude via WayLog_


**User**

add script of the bottom to find bounding boxes for ships.

---


**User**

[Request interrupted by user]

---


**User**

image.png
630×592
c:\Users\owner\Downloads\supply-chain-satellite-ai-main\supply-chain-satellite-ai-main\satellite_supply_chain_model (1).ipynb
add script of the bottom to find bounding boxes for ships.

---


**Claude**

New cell id is `cell-11`. Now I'll add the code cell with the reusable detection function right after it.

---


**Claude**

Added two cells at the bottom of the notebook: a markdown header and a `detect_ships(image_path, model, ...)` function that wraps the existing water-mask + CNN sliding-window + NMS pipeline (from the "Testing" cell) into a single reusable call.

It returns a dict with `boxes` (list of `[x1, y1, x2, y2, confidence]`), the scaled ROI image, the drawn output image, and the water mask — so you can now run detection on any image/model without duplicating the pipeline. An example call at the end runs it on `port_area_cropped.jpg` and plots the result. Since this is a Colab notebook, run it there to verify.

---
