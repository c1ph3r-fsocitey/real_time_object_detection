from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
results = model.predict(source=0, show=True)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

prev_time = time.time()
frame_count = 0
start_time = prev_time  # Added to calculate elapsed time for average FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time
    frame_count += 1

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop ends, calculate the average FPS
avg_fps = frame_count / (time.time() - start_time)

# Print the average FPS
print(f'Average FPS: {avg_fps:.2f}')

cap.release()
out.release()
cv2.destroyAllWindows()
