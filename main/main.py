import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('frontalface_default.xml')
specs_ori = cv2.imread('think.png', -1)
#smile_ori = cv2.imread("smile.png")

cap = cv2.VideoCapture(0) #webcame video
# cap = cv2.VideoCapture('jj.mp4') #any Video file also
cap.set(cv2.CAP_PROP_FPS, 60)



def transparentOverlay(src, overlay):
    h, w, _ = overlay.shape  # Size of foreground

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            alpha = overlay[i][j][3] # read the alpha channel
            if alpha == 255:
                src[x + i][y + j] = overlay[i][j][:3]
    return src

while 1:
    ret, img = cap.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:

            glass_symin = int(y + 0.5 * h / 5)
            glass_symax = int(y + 3.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            face_part = img[y:y+h, x:x+w]

            specs = cv2.resize(specs_ori, (w, h),interpolation=cv2.INTER_CUBIC)
            #transparentOverlay(face_glass_roi_color,specs)
            
            #smile = cv2.resize(smile_ori, (100, 100))
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            #img[y:y+h, x:x+w] = specs
            alphas = np.stack((specs[:, :, 3], specs[:, :, 3], specs[:, :, 3]), axis = 2)//255
            alphasb = np.stack((255 - specs[:, :, 3], 255 - specs[:, :, 3], 255 - specs[:, :, 3]), axis = 2)//255
            img[y:y+h, x:x+w] = specs[:, :, :3] * alphas + face_part * alphasb
            
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))

    cv2.imshow('Project', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('img.jpg', img)
        break

cap.release()

cv2.destroyAllWindows()