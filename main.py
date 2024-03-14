import cv2
import easyocr
import matplotlib.pyplot as plt
#read image
image_path = '/Users/ryaanmohideen/text detection/covid sign.jpeg'
img = cv2.imread(image_path)

#instance text detector
reader = easyocr.Reader(['en'], gpu=False)

#detect text on image

text_og  = reader.readtext(img)
threshold = 0.8
# draw box and text
for t in text_og:
    print(t)
    
    box, text, score = t

    if (score > threshold):
        cv2.rectangle(img, box[0], box[2], (0,260,0), 8)
        cv2.putText(img, text, box[0], cv2.FONT_HERSHEY_DUPLEX, 1, (260, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
