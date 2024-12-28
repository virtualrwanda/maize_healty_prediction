from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start_preview()
picam2.configure(picam2.create_still_configuration())
picam2.start()

frame = picam2.capture_array()
cv2.imshow("Camera Frame", frame)
cv2.waitKey(0)

picam2.close()
cv2.destroyAllWindows()

