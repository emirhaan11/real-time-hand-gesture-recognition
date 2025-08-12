# Finger Counting with OpenCV

This project demonstrates a **real-time finger counting application** using **Python**, **OpenCV**, and **NumPy**. It detects a user’s hand from the webcam feed, processes the image to isolate the hand region, and counts the number of raised fingers.

## 📌 Features
- **Real-time hand detection** using webcam.
- **Background subtraction** and skin-color segmentation for hand isolation.
- **Convex hull & defect analysis** to detect and count raised fingers.
- Easy-to-read and customizable Python code.
- Runs in a Jupyter Notebook for step-by-step demonstration.

## 🛠️ Technologies Used
- **Python 3.x**
- [OpenCV](https://opencv.org/) — Image processing and computer vision tasks.
- [NumPy](https://numpy.org/) — Efficient numerical computations.
- [Jupyter Notebook](https://jupyter.org/) — Interactive code execution.

## 🚀 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/emirhaan11/real-time-hand-gesture-recognition
   cd real-time-hand-gesture-recognition

2. **Install Dependencies**
   ```bash
   pip install opencv-python numpy scikit-learn

2. **Run the notebook**
   ```bash
   jupyter notebook finger_count.ipynb

 ## 📖 How to Use
 1. Make sure your webcam is connected and working.
 2. Run all cells in the Jupyter Notebook.
 3. When prompted, place your hand in front of the camera within the detection frame.
 4. For best results, keep your hand in front of a plain, pattern-free background such as a flat wall. This reduces noise and improves detection accuracy.
 5. Hold your fingers apart for better detection accuracy.
 6. The application will display the number of fingers it detects in real time.
 7. Press **esc** to quit the webcam window.
 





