#  Self-Driving Car Object Detection using DETR Model

This project focuses on building a high-performance **Self-Driving Car Object Detection System** capable of detecting multiple road objects in real time. The model was trained on a large-scale dataset containing **30,000+ annotated images** covering **10 different classes**, making it suitable for autonomous driving and traffic analytics applications.

The system identifies critical road objects including:

-  Car  
-  Truck  
-  Motorcycle  
-  Bicycle  
-  Traffic Lights  
-  Traffic Signs  
-  Pedestrians  
-   Bus  
-  Ambulance  
-  Two-wheelers 

#  **Key Features**

- ✔️ Trained on **30k+ labeled images**
- ✔️ Supports **10 real-world driving classes**
- ✔️ Real-time object detection (video & image)
- ✔️ Detection using **Deformable DETR** / **YOLOv8**
- ✔️ Fully optimized preprocessing & inference pipeline
- ✔️ Streamlit interface for easy testing
- ✔️ Robust detection in highways, traffic, and urban environment
- 
# **Model Architecture**

This project uses **Deformable DETR** (by SenseTime) and optionally YOLO models for fast and accurate detection.

### Why Deformable DETR?
- Better performance on dense scenes  
- Handles long-range dependency  
- High accuracy for small objects  
- Faster convergence than vanilla DETR  

### Model Pipeline:
1. Image resize + normalization  
2. Feature extraction with transformer backbone  
3. Attention-based bounding box prediction  
4. Non-max suppression (NMS)  
5. Label & confidence assignment
6. 
# 📊 **Dataset**

### **Total Images**: 30,000+  
### **Classes (10)**:


Dataset included highway, urban, and mixed driving scenarios with varying lighting and weather conditions.
# deployment link
https://selfdrivingobjectdetection.streamlit.app/

#  **Detection Results**

![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/self_1.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/self_1.1.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/car_2.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/car_2.2.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/car_3.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/self_driving/blob/main/car_3.3.jpg?raw=true)


