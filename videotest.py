import numpy as np
import json
import cv2
import keras

jsonFileName='detect.json'
model=keras.models.model_from_json(json.load(open(jsonFileName,'r')))
model.load_weights('model.hdf5')
print("done loading weights of trained model")

labels = ['healthy','Raw','unhealthy']

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imshow('input frame',frame)
        cropped = cv2.resize(frame,(200,200))
        cropped= cropped/255
        cropped = np.reshape(cropped,[1,200,200,3])
        classes = model.predict(cropped)
        output = np.argmax(classes)
        
        print(np.argmax(classes))
        print(labels[output])
                    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, labels[output], (30,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('output frame',frame)
        cv2.waitKey(1)
    cv2.waitKey(1)    
    
    cv2.imshow('input',frame)
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
