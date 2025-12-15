import cv2
import time
import torch
import numpy as np

#MiDaS model type
model_type = 'MiDaS_small'

# Loading MiDaS
midas = torch.hub.load('intel-isl/MiDaS',model_type)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Transform to relevant device
midas.to(device)

midas.eval()

# Load transforms to apply resize and normalize the images
midas_transforms = torch.hub.load('intel-isl/MiDaS',"transforms")

# load relevant transforms
if model_type == "DPT_Large" or model_type == "DPT_hybrid":
    transforms = midas_transforms.dpt_transform
else:
    transforms = midas_transforms.small_transform

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success,img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    input_batch = transforms(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map,cv2.COLORMAP_COOL)

        cv2.putText(img,f'FPS : {int(fps)}' , (20,70), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)
        cv2.imshow('MiDaS',img)
        cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






