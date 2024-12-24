# Sign Language Detection  

## Table of Contents  
- [About](#about)  
- [Technologies](#technologies)  
- [Prerequisites](#prerequisites)  
- [Features](#features)  
- [Demo](#demo)  
- [Issues and Next Steps](#issues-and-next-steps)  

---

## About  
This project aims to detect and classify **sign language gestures** using **machine learning** and **computer vision**.  

---

## Technologies  
- **Python 3.8+**  
- **OpenCV**  
- **MediaPipe**  
- **Scikit-Learn**  

---
## Features

✨ Hand Landmark Detection using MediaPipe.
✨ Multi-Class Classification with Random Forest Classifier.
✨ Custom Dataset Collection for training gestures.

---
## Demo
### Landmark Detection Example
<p align="center">
  <img src="./readme_media/landmark1.png" width="300"/>
  <img src="./readme_media/landmark2.png" width="300"/>
  <img src="./readme_media/landmark3.png" width="300"/>
</p>

### Model Prediction Example

<p align="center">
  <img src="./readme_media/failure_example.gif" width="500"/>
</p>

---
## Issues 🚧

### Current Issue:
	•	The model often predicts **'your'** for most gestures, failing to differentiate between inputs.
