"""
Demo script để test visualization utilities
"""
import numpy as np
import os
from utils.visualization import (
    visualize_predictions
)

def create_dummy_data(batch_size=4, his_len=10, pred_len=10, num_channels=3, height=32, width=32):
    """Tạo dữ liệu giả để test visualization"""
    # Tạo historical data
    inputs = np.random.randn(batch_size, his_len, num_channels, height, width) * 0.5 + 0.5
    
    # Tạo predictions (có một số noise)
    preds = np.random.randn(batch_size, pred_len, num_channels, height, width) * 0.5 + 0.5
    
    # Tạo ground truth (tương tự predictions nhưng với pattern khác)
    trues = np.random.randn(batch_size, pred_len, num_channels, height, width) * 0.5 + 0.5
    
    # Thêm trend để dễ nhìn trên biểu đồ đường
    for b in range(batch_size):
        for c in range(num_channels):
            # Tạo một đường cong cơ bản
            base_curve = np.sin(np.linspace(0, 3*np.pi, his_len + pred_len))
            
            # Gán giá trị trung bình cho các frame
            for t in range(his_len):
                inputs[b, t, c] += base_curve[t]
            
            for t in range(pred_len):
                trues[b, t, c] += base_curve[his_len + t]
                preds[b, t, c] += base_curve[his_len + t] + np.random.normal(0, 0.2) # Pred lệch một chút
    
    return inputs, preds, trues


def main():
    print("="*60)
    print("Testing Visualization Utilities")
    print("="*60)
    
    # Tạo dữ liệu test
    print("\n1. Creating dummy data...")
    inputs, preds, trues = create_dummy_data(batch_size=4, his_len=10, pred_len=10, num_channels=3, height=32, width=32)
    print(f"   Inputs shape: {inputs.shape}")
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Ground truth shape: {trues.shape}")
    
    # Tạo thư mục output
    output_dir = './test_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test function tổng hợp
    print("\n2. Testing comprehensive visualization function...")
    visualize_predictions(inputs, preds, trues, output_dir, prefix='demo')
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
