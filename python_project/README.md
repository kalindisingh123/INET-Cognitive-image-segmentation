# AI-Based Skin Cancer Detection and Tumor Spread Analysis System

## System Architecture
The system employs a dual-model approach:
1. **Segmentation (U-Net):** A deep convolutional neural network designed for biomedical image segmentation. It consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization.
2. **Classification (ResNet50):** A state-of-the-art residual network used via Transfer Learning. It classifies the detected region into Benign or Malignant categories.

## Spread Calculation
The spread percentage is calculated as:
`Spread % = (Number of Tumor Pixels / Total Number of Pixels in Image) * 100`
This provides a quantitative measure of the tumor's physical extent on the skin surface.

## Advantages
- **Early Detection:** Helps in identifying suspicious lesions early.
- **Quantitative Analysis:** Provides exact spread percentage, reducing human error.
- **Non-Invasive:** Preliminary analysis without biopsy.

## Limitations
- **Data Dependency:** Accuracy depends heavily on the quality and diversity of the training dataset (ISIC/HAM10000).
- **Surface Level:** Only analyzes surface spread, not depth (Breslow thickness).
- **Lighting/Artifacts:** Sensitive to image quality, hair, and lighting conditions.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare dataset in `data/` folder.
3. Run `train.py` to train models.
4. Use `predict.py` for inference on new images.
5. Refer to `optimization_report.md` for detailed performance tuning findings.
