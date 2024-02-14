from ultralytics import YOLO 
model = YOLO(r'E:\desk\tooth-2d\yolov8n-seg.pt')  
results = model.train(data = 'tooth_2d_Seg.yaml', 
                      epochs = 10, 
                      workers = 0, 
                      batch = -1, 
                      seed = 888)

