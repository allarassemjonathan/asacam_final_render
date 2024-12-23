### Prerequisites
- Python 3.7.6
- CUDA 11.7
- NVIDIA GPU 

### Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/asamaandrone/productx.git
   cd productx
   ```

2. **Install Dependencies**
   Ensure you have the required dependencies by running:
   ```sh
   pip install -r requirements_yolox.txt
   ```
   If you run into this error:
   ```sh
   ERROR: Could not find a version that satisfies the requirement torch==1.13.1+cu117 (from versions: 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1)
   ERROR: No matching distribution found for torch==1.13.1+cu117
   ```
   Run this line:
   ```sh
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```
   Then, re-run the dependencies installation with this line:
   ```sh
   pip install -r requirements_yolox.txt
   ```


4. **Download YOLOX Model Weights**
   Download the YOLOX model weights for boat detection from [this Google Drive link](<https://drive.google.com/drive/u/0/folders/10lWeA-j1VY55KOyfs285phT1Slwu2y7V>).

   Once downloaded, place the weights file (`best_ckpt.pth`) within the `YOLOX_outputs/yolox_l` directory.

6. **Download VIT-GPT2 Model Weights**
   Download the VIT-GPT2 model weights from [this Google Drive link](<https://drive.google.com/drive/u/0/folders/13xbo4Kevm8LY4WNsyHnf279tyQnYrD4C>).

   Once downloaded, place the weights file (`pytorch_model.bin`) within the `tools/vit-gpt2-image-captioning` directory.

7. **Run the YOLOX Demo**
   To run the YOLOX demo on your video, use the following command:
   ```sh
   python tools/demo.py video -n yolox-l -c YOLOX_outputs/yolox_l/best_ckpt.pth --path "path to your video" --conf 0.75 --nms 0.45 --tsize 640 --device gpu
   ```

   Replace `"path to your video"` with the actual path to your video file.

8. **Run the YOLOX-OPENAI Demo**
   To run the YOLOX-OPENAI demo on your video, use the following command:
   ```sh
   python tools/demo_fm.py video -n yolox-l -c YOLOX_outputs/yolox_l/best_ckpt.pth --path "path to your video" --conf 0.75 --nms 0.45 --tsize 640 --device gpu
   ```

   Replace `"path to your video"` with the actual path to your video file.

9. **Run the YOLOX-VIT-GPT2 Demo**
   To run the YOLOX-VIT-GPT2 demo on your video, use the following command:
   ```sh
   python tools/demo_vit.py video -n yolox-l -c YOLOX_outputs/yolox_l/best_ckpt.pth --path "path to your video" --conf 0.75 --nms 0.45 --tsize 640 --device gpu
   ```

   Replace `"path to your video"` with the actual path to your video file.

10. **Fix the display issue of YOLOX**
   Issue: boats are detected as 'person'
   Follow the video in this [link](<https://drive.google.com/drive/u/0/folders/1FdRsUP5RxL6Ym_9LL-qPamUYV_q7Fh52>) to resolve the issue. 
