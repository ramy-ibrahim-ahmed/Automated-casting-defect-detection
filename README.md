# Automated Casting Defect Detection

**Project Overview**
This initiative seeks to transform the casting manufacturing industry by automating the defect detection process. Traditionally, this task relies on manual inspection, which is not only time-consuming and labor-intensive but also susceptible to human error. To overcome these limitations, I have developed and implemented a deep learning-based solution using state-of-the-art architectures: ResNet-18, ResNet-50, and a fine-tuned Inception V3 model.

**Impact and Benefits**  
This solution enhances manufacturing efficiency by:

- Minimizing manual inspection errors.  
- Accelerating the defect detection process.  
- Enabling early identification of flaws to reduce production waste and costs.  

---

## ResNet-18

![image 1](results/ResNet-18/accuracy_loss_plot.png)

![image 2](results/ResNet-18/precision_recall_f1_plot.png)

![image 3](results/ResNet-18/threshold/roc_and_confusion_th=0.1.png)

---

## ResNet-50

![image 1](results/ResNet-50/accuracy_loss_plot.png)

![image 2](results/ResNet-50/precision_recall_f1_plot.png)

![image 3](results/ResNet-50/threshold/roc_and_confusion_th=0.1.png)

---

## Inception-V3

![image 1](results/Inception-v3/accuracy_loss_plot.png)

![image 2](results/Inception-v3/precision_recall_f1_plot.png)

![image 3](results/Inception-v3/roc_and_confusion_th=0.4.png)

---

## Deployment  

To ensure a seamless and efficient deployment process, I utilized modern development tools and frameworks:  

**REST--APIs**  
   The trained models (ResNet-18, ResNet-50, and fine-tuned Inception V3) were deployed using **FastAPI**, enabling a robust and high-performance API for defect detection.  

**Containerization**  
The application was fully containerized using Docker and Docker Compose, ensuring:  

- Portability across environments.  
- Simplified deployment and scaling.  
- Streamlined dependency management.

![API](results/API.png)

---
