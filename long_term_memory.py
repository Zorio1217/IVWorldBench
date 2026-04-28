import argparse
import cv2
from torchmetric.psnr_metrics import PeakSignalNoiseRatioMetric
from torchmetric.lpips_metrics import LearnedPerceptualImagePatchSimilarityMetric
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate long-term memory performance of a model.')
parser.add_argument('--video_path', type=str, required=True, help='Path to the video file to be evaluated.')

def extract_symmetric_pairs(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    # 读取所有帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    n = len(frames)
    pairs = []

    for i in range(n):
        pairs.append((frames[i], frames[n - 1 - i]))

    return pairs

def evaluate_long_term_memory(video_path):
    print(f"Evaluating long-term memory performance on video: {video_path}")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    pairs = extract_symmetric_pairs(video_path)
    psnr_scores = []
    lpips_scores = []
    
    # 计算 PSNR 和 LPIPS
    for pair in pairs:
        first_frame, last_frame = pair
        psnr_metric = PeakSignalNoiseRatioMetric()
        lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()
        psnr_score = psnr_metric._compute_scores(first_frame, last_frame)
        lpips_score = lpips_metric._compute_scores(first_frame, last_frame)
        psnr_scores.append(psnr_score)
        lpips_scores.append(lpips_score)
    psnr_scores = np.array(psnr_scores)
    psnr_scores = np.clip(psnr_scores, 0, 100)  # 将 PSNR 分数限制在合理范围内
    lpips_scores = np.array(lpips_scores)
    print(f"PSNR Score: {np.mean(psnr_scores):.2f} dB")
    print(f"LPIPS Score: {np.mean(lpips_scores):.4f}")
    plt.plot(psnr_scores[:len(psnr_scores)//2], label='PSNR')
    plt.plot(lpips_scores[:len(lpips_scores)//2], label='LPIPS')
    plt.xlabel('Frame Index')
    plt.ylabel('Score')
    plt.title('Long-Term Memory Performance')
    plt.legend()
    plt.savefig('long_term_memory_performance.png')

if __name__ == "__main__":
    args = parser.parse_args()
    evaluate_long_term_memory(args.video_path)
    