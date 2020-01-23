import numpy as np
import cv2
# from imutils.video import VideoStream
# from imutils import resize
import time
import os
import glob
import pdb


import torch
from torchvision import transforms
from utils import get_config
from trainer import Trainer
from PIL import Image


cap = cv2.VideoCapture(0)
cv2.namedWindow("Output")

# face detector
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile('haarcascade_frontalface_alt.xml')):
    print('--(!)Error loading face cascade')
    exit(0)

# alpha mask
mask = cv2.imread("alpha_mask2.png")
mask = np.float32(mask)/255
# mask = mask * 255

# Calculate class codes for specified image
ckpt = 'pretrained/animal149_gen.pt'
class_image_folder = 'images/golden_retrievers'


config = get_config('configs/funit_animals.yaml')
config['batch_size'] = 1
config['gpus'] = 1

trainer = Trainer(config)
trainer.load_ckpt(ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)

print('Compute average class codes for images in %s' % class_image_folder)

# obtain all images with .jpg extension
class_images = [file for file in os.listdir(class_image_folder) if file.endswith(".jpg")]
for i, f in enumerate(class_images):
    fn = os.path.join(class_image_folder, f)

    print('filname', fn)

    img = Image.open(fn).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(img_tensor, 1)
        if i == 0:
            new_class_code = class_code
        else:
            new_class_code += class_code
final_class_code = new_class_code / len(class_images)

# by how much to expand the face ROI
k = 0.3

while True:
    # preparation(vs,prep_time,rotate)
    ret, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    # frame_pil = Image.fromarray(np.uint8(frame))
    # input_img = transform(frame_pil).unsqueeze(0)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect multiple faces
    faces = face_cascade.detectMultiScale(frame_gray)

    has_face = False

    x1 = x2 = y1 = y2 = 0
    for (x,y,w,h) in faces:
        # only use faces with reasonable width/height
        if len(faces) > 1 and w < 100 and h < 100:
            continue
        # use 1.25x the detected rectangle
        x1 = int(x-k*w)
        y1 = int(y-k*h)

        x2 = int(x+(1+k)*w)
        y2 = int(y+(1+k)*h)


        has_face = x1 > 0 and y1 > 0 and x2 < width and y2 < height and w > 0 and h > 0
        print('Detected face? ', has_face)

        # crop frame
        if has_face:
            # draw rect over face
            # frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 4)

            face_frame = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(np.uint8(face_frame))
            face_input = transform(face_pil).unsqueeze(0)


    if has_face:
        with torch.no_grad():
            translation = trainer.model.translate_simple(face_input, final_class_code)
            output = translation.detach().cpu().squeeze().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = np.float32((output + 1) * 0.5)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # scale mask and translation
            scaled_mask = cv2.resize(mask, (x2-x1, y2-y1))
            scaled_output = cv2.resize(output, (x2-x1, y2-y1))

            # Multiply the translated image with the alpha matte
            masked_translation = cv2.multiply(scaled_mask, scaled_output)

            # Multiply the face with ( 1 - alpha )
            frame = np.float32(frame)/255
            masked_face_region = cv2.multiply(1.0 - scaled_mask, frame[y1:y2, x1:x2])

            # # Add the masked face and translation
            masked_output = cv2.add(masked_translation, masked_face_region)

            # print(masked_output[64])

            # overlay output onto video stream
            frame[y1:y2, x1:x2] = masked_output


            cv2.imshow('Output', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break



