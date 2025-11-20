import numpy as np
import matplotlib.pyplot as plt
import os

def plot_series_visualization(historical, preds, trues, sample_idx=0, channel_idx=0, save_path=None):
    hist_mean = historical.mean(axis=(3, 4))
    pred_mean = preds.mean(axis=(3, 4))
    true_mean = trues.mean(axis=(3, 4))

    hist_series = hist_mean[sample_idx, :, channel_idx]
    pred_series = pred_mean[sample_idx, :, channel_idx]
    true_series = true_mean[sample_idx, :, channel_idx]
    
    his_len = len(hist_series)
    pred_len = len(pred_series)
    total_len = his_len + pred_len
    
    # Tạo trục thời gian
    t_hist = np.arange(his_len)
    t_future = np.arange(his_len - 1, total_len) # Bắt đầu từ điểm cuối của history để nối liền
    
    last_hist_val = hist_series[-1]
    pred_series_plot = np.concatenate(([last_hist_val], pred_series))
    true_series_plot = np.concatenate(([last_hist_val], true_series))
    
    plt.figure(figsize=(12, 6))
    
    # Vẽ History (Nửa bên trái)
    plt.plot(t_hist, hist_series, label='Historical', color='blue', linewidth=2, marker='o', markersize=4)
    
    # Vẽ Ground Truth (Nửa bên phải)
    plt.plot(t_future, true_series_plot, label='Ground Truth', color='green', linewidth=2, linestyle='-', marker='s', markersize=4)
    
    # Vẽ Prediction (Nửa bên phải)
    plt.plot(t_future, pred_series_plot, label='Prediction', color='red', linewidth=2, linestyle='--', marker='x', markersize=4)
    
    # Thêm đường phân cách
    plt.axvline(x=his_len - 1, color='gray', linestyle=':', alpha=0.7)
    plt.text(his_len - 1, plt.ylim()[1]*0.95, 'Forecast Start', ha='center', va='top', backgroundcolor='white')
    
    plt.title(f'Time Series Forecast (Sample {sample_idx}, Channel {channel_idx})', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved series plot to {save_path}")
    
    plt.close()

def visualize_predictions(historical, preds, trues, output_dir, prefix='test'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating visualizations in: {output_dir}")
    print(f"{'='*60}")
    
    num_channels = preds.shape[2]
    
    # Vẽ biểu đồ cho channel đầu tiên của sample đầu tiên
    print("Creating series visualization for Sample 0, Channel 0...")
    plot_series_visualization(
        historical, preds, trues,
        sample_idx=0, channel_idx=0,
        save_path=os.path.join(output_dir, f'{prefix}_series_sample0_ch0.png')
    )
    
    # Vẽ thêm một vài sample khác để kiểm tra
    if historical.shape[0] > 1:
        print("Creating series visualization for Sample 1, Channel 0...")
        plot_series_visualization(
            historical, preds, trues,
            sample_idx=1, channel_idx=0,
            save_path=os.path.join(output_dir, f'{prefix}_series_sample1_ch0.png')
        )
        
    # Vẽ cho tất cả các channels của sample 0
    for ch in range(min(num_channels, 3)): # Giới hạn 3 channels đầu
        if ch == 0: continue # Đã vẽ ở trên
        print(f"Creating series visualization for Sample 0, Channel {ch}...")
        plot_series_visualization(
            historical, preds, trues,
            sample_idx=0, channel_idx=ch,
            save_path=os.path.join(output_dir, f'{prefix}_series_sample0_ch{ch}.png')
        )
    
    print(f"{'='*60}")
    print(f"Visualization completed!")
    print(f"{'='*60}\n")
