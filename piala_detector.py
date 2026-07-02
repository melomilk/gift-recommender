import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# load CLIP
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model loaded. Press SPACE to capture, Q to quit.")

# labels to detect — add or remove anything you want
labels = [
    "a piala, traditional Kazakh ceramic tea bowl",
    "a dombra, traditional Kazakh string instrument",
    "a keseshka, Kazakh tea cup",
    "a platok, Kazakh embroidered shawl",
    "a regular coffee mug",
    "a water bottle",
    "a book",
    "a phone",
    "a keyboard",
    "a person",
    "nothing in particular"
]

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't access webcam")
        break

    # show instructions on screen
    cv2.putText(frame, "SPACE = detect  |  Q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("CLIP Object Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        # capture current frame and run CLIP
        print("\nAnalyzing...")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(
            text=labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)

        # print top 3 results
        top3 = probs.topk(3)
        print("\n--- CLIP sees: ---")
        for score, idx in zip(top3.values, top3.indices):
            print(f"  {labels[idx]} → {score.item()*100:.1f}%")
        print("------------------")

cap.release()
cv2.destroyAllWindows()