# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2

import tensorflow as tf

dd = {0: 'C2', 1: 'C3', 2: 'C4', 3: 'C5', 4: 'C6', 5: 'D2', 6: 'D3', 7: 'D4', 8: 'D5', 9: 'D6', 10: 'H2', 11: 'H3', 12: 'H4', 13: 'H5', 14: 'H6', 15: 'S2', 16: 'S3', 17: 'S4', 18: 'S5', 19: 'S6'}

new_model = tf.keras.models.load_model('model.h5')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(0.5)
fps = FPS().start()

h = 200 # 355
w = 200
dim = (h,w)    

# loop over the frames from the video stream
frame_id = 0
while True:
    frame = vs.read()   
    # frame_id += 1
    # filename = 'frames/frame_' + str(frame_id) + '.png'
    # cv2.imwrite(filename, frame)

    # show the output frame
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # BGR to GRAY
    frame_rs = cv2.resize(img_g, dim, interpolation=cv2.INTER_AREA) # Изменение размера
    pred = []
    pred.append(frame_rs)
    pred = np.asarray(pred) 
    pred = pred.astype('float32')   
    pred = pred.reshape(pred.shape[0], h, w, 1)
    result = new_model.predict_classes(pred)
    predictions = new_model.predict(pred)

    cv2.imshow("Frame", frame_rs)
    key = cv2.waitKey(1) & 0xFF

    if (np.max(predictions) > 0.6):
        print('-----------------------')
        print(result)
        res = dd.get(result[0])
        if (res[0] == 'C'):            
            print("Kresti", dd.get(result[0]) )
        if (res[0] == 'D'):            
            print("Bubi", dd.get(result[0]))
        if (res[0] == 'H'):            
            print("4epBu", dd.get(result[0]))
        if (res[0] == 'S'):            
            print("Vinni", dd.get(result[0]))
        print(np.max(predictions))

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    time.sleep(0.1)
    # update the FPS counter
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()