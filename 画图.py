import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import numpy as np

# 设置绘图风格，使用更现代、精美的样式
plt.style.use('seaborn-v0_8-whitegrid')
# 设置字体，确保兼容性 (Times New Roman 是学术论文常用字体)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_financial_data(csv_file):
    """
    读取 CSV 文件并生成精美的多面板学术图表。
    """
    try:
        # 读取数据
        df = pd.read_csv(csv_file)

        # 1. 数据清洗与预处理
        # 转换 Time 列为 datetime 对象
        df['Time'] = pd.to_datetime(df['Time'])

        # 排序 (虽然通常是排好的，但保险起见)
        df = df.sort_values('Time')

        # 2. 数据平滑处理 (可选，为了曲线更美观)
        # 使用 Savitzky-Golay 滤波器进行平滑，同时保留数据特征
        # window_length 必须是奇数，根据数据量调整
        window = min(51, len(df) // 5 | 1)
        if window > 3:
            df['AUM_Smooth'] = savgol_filter(df['AUM'], window, 3)
            df['Equity_Smooth'] = savgol_filter(df['Equity'], window, 3)
            df['Group_Cash_Smooth'] = savgol_filter(df['Group_Cash'], window, 3)
        else:
            df['AUM_Smooth'] = df['AUM']
            df['Equity_Smooth'] = df['Equity']
            df['Group_Cash_Smooth'] = df['Group_Cash']

        # 3. 创建画布布局
        # 使用 GridSpec 创建非均匀布局：上面一个大图，下面两个小图
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)

        # --- 子图 1: 核心资产曲线 (占据顶部两行) ---
        ax1 = fig.add_subplot(gs[0, :])

        # 绘制原始数据 (浅色透明) 和平滑数据 (深色实线)
        ax1.plot(df['Time'], df['AUM'], color='#A6CEE3', alpha=0.4, linewidth=1)
        ax1.plot(df['Time'], df['AUM_Smooth'], color='#1F78B4', linewidth=2.5, label='Total AUM (Net Worth)')

        ax1.plot(df['Time'], df['Equity'], color='#B2DF8A', alpha=0.4, linewidth=1)
        ax1.plot(df['Time'], df['Equity_Smooth'], color='#33A02C', linewidth=2, linestyle='--',
                 label='Agent Equity (Holdings)')

        # 标注关键事件 (Liquidation Cascade)
        # 找到 Group_Cash 剧烈变化的点 (22:53)
        crash_time = pd.to_datetime('2025-12-05 22:53:00')
        # 确保时间在数据范围内
        if df['Time'].min() <= crash_time <= df['Time'].max():
            # 在图上标记
            # 获取该时刻的 AUM 值
            crash_val = df.loc[df['Time'] >= crash_time, 'AUM'].iloc[0]

            ax1.annotate('Liquidation Cascade\n(Systemic Collapse)',
                         xy=(crash_time, crash_val),
                         xytext=(crash_time, crash_val + 15000),
                         arrowprops=dict(facecolor='#E31A1C', shrink=0.05, width=2, headwidth=8),
                         fontsize=12, fontweight='bold', color='#E31A1C', ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E31A1C", alpha=0.8))

            # 画一条竖直虚线
            ax1.axvline(x=crash_time, color='#E31A1C', linestyle=':', alpha=0.6)

        # 填充亏损区域 (基准线 50000)
        ax1.fill_between(df['Time'], df['AUM_Smooth'], 50000,
                         where=(df['AUM_Smooth'] < 50000),
                         color='#FB9A99', alpha=0.2, label='Capital Erosion')

        ax1.set_ylabel('USD Value', fontweight='bold')
        ax1.set_title('A. Systemic Capital Decay Dynamics', loc='left', fontweight='bold', fontsize=14)
        ax1.legend(loc='lower left', frameon=True, framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 格式化 X 轴 (时间)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # --- 子图 2: 系统流动性 (左下) ---
        ax2 = fig.add_subplot(gs[1, 0])

        # 绘制 Group Cash (这是体现 Soft Budget Constraint 的关键)
        ax2.plot(df['Time'], df['Group_Cash'], color='#E31A1C', alpha=0.3, linewidth=1)
        ax2.plot(df['Time'], df['Group_Cash_Smooth'], color='#E31A1C', linewidth=2, label='System Cash (Liquidity)')

        # 强调 0 轴
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')

        # 标注赤字区域
        ax2.fill_between(df['Time'], df['Group_Cash_Smooth'], 0,
                         where=(df['Group_Cash_Smooth'] < 0),
                         color='#E31A1C', alpha=0.1)

        ax2.set_ylabel('USD', fontweight='bold')
        ax2.set_title('B. Liquidity Crisis (Soft Budget Constraint)', loc='left', fontweight='bold', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # --- 子图 3: 投资回报率 (右下) ---
        ax3 = fig.add_subplot(gs[1, 1])

        # ROI 曲线
        roi_vals = df['ROI'] * 100  # 转换为百分比
        ax3.plot(df['Time'], roi_vals, color='#FF7F00', linewidth=2, label='ROI (%)')

        ax3.set_ylabel('Percentage (%)', fontweight='bold')
        ax3.set_title('C. Return on Investment (ROI)', loc='left', fontweight='bold', fontsize=12)

        # 颜色映射背景 (绿色=盈利, 红色=亏损)
        # 这里全亏，所以其实只有红色背景
        ax3.axhspan(-100, 0, color='#FDBF6F', alpha=0.1)
        ax3.axhspan(0, 50, color='#B2DF8A', alpha=0.1)

        ax3.grid(True, linestyle=':', alpha=0.5)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # --- 子图 4: 亏损速度 (Drawdown Rate) (底部通栏) ---
        # 计算每分钟的 AUM 变化 (一阶导数)
        # 为了平滑，使用 rolling mean
        df['Drawdown_Rate'] = df['AUM'].diff().fillna(0)
        df['Drawdown_Rate_Smooth'] = df['Drawdown_Rate'].rolling(window=10).mean()

        ax4 = fig.add_subplot(gs[2, :])

        # 柱状图展示
        colors = np.where(df['Drawdown_Rate_Smooth'] >= 0, '#33A02C', '#E31A1C')
        ax4.bar(df['Time'], df['Drawdown_Rate_Smooth'], width=0.0005, color=colors, alpha=0.6, label='Minute PnL')

        # 强调崩盘时刻的巨大负值
        if df['Time'].min() <= crash_time <= df['Time'].max():
            ax4.axvline(x=crash_time, color='black', linestyle='--', alpha=0.5)
            ax4.text(crash_time, df['Drawdown_Rate'].min() * 0.8, ' Flash Crash', color='black',
                     verticalalignment='center')

        ax4.set_ylabel('Minute PnL ($)', fontweight='bold')
        ax4.set_title('D. Volatility of Loss (Minute-by-Minute PnL)', loc='left', fontweight='bold', fontsize=12)
        ax4.grid(True, linestyle=':', alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.set_xlabel('Simulation Time (UTC)', fontsize=12, fontweight='bold')

        # 4. 全局美化与保存
        plt.suptitle('Figure 1: The Anatomy of Failure - Empirical Results from "Galaxy Empire"',
                     fontsize=16, fontweight='bold', y=0.95)

        # 自动调整标签防止重叠
        plt.gcf().autofmt_xdate()

        # 保存为高分辨率图片 (PDF 适合论文, PNG 适合预览)
        plt.savefig('galaxy_empire_results.png', dpi=300, bbox_inches='tight')
        plt.savefig('galaxy_empire_results.pdf', bbox_inches='tight')

        print("绘图完成！已保存为 galaxy_empire_results.png 和 galaxy_empire_results.pdf")
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
    except Exception as e:
        print(f"绘图过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用你提供的文件名
    csv_filename = 'finance_log_v67.csv'

    # 为了演示，你可以直接运行这个脚本，如果文件在同一目录下
    plot_financial_data(csv_filename)