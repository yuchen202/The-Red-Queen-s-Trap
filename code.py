import os
import asyncio
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, Canvas, messagebox
from collections import deque, defaultdict, Counter
import websockets
import time
import random
import numpy as np
import threading
import queue
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import ssl
import math
import warnings
import urllib.request
import urllib.error
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glob
from torch.utils.data import Dataset, DataLoader
import gc
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==================================================================================
# 1. 银河帝国 V65 - 旗舰监控版配置
# ==================================================================================
CONFIG = {
    "TARGET_COINS": [
        'btcusdt', 'ethusdt', 'bnbusdt', 'solusdt', 'xrpusdt', 'dogeusdt',
        'adausdt', 'avaxusdt', 'trxusdt', 'dotusdt', 'linkusdt', 'maticusdt',
        'ltcusdt', 'bchusdt', 'shibusdt', 'nearusdt', 'pepeusdt', 'filusdt',
        'atomusdt', 'uniusdt'
    ],
    "API_SOURCES": ["https://api.binance.com/api/v3/klines", "https://api-gcp.binance.com/api/v3/klines"],
    "WSS_URL": "wss://stream.binance.com:9443",
    "DATA_DIR": Path("./backtest_data_v67"),
    "TEMP_DIR": Path("./temp_realtime_data"),
    "CHECKPOINT_DIR": Path("./checkpoints_v67"),
    "FINANCE_LOG": "finance_log_v67.csv",

    "AGENT_COUNT": 500,
    "INIT_CAPITAL": 100.0,
    "INTERVAL": "15m",
    "DOWNLOAD_DAYS": 60,
    "MIN_APY_FOR_LIVE": 0.20,
    "TRANSACTION_FEE": 0.0004,  # 开/平仓手续费 (万4)
    "SLIPPAGE": 0.0002,  # 滑点 (万2)
    "LEVERAGE_INTEREST": 0.000004,  # 杠杆借贷每小时利息 (万5/小时，这在币圈算正常偏高，高杠杆杀手)

    "SEQ_LEN": 60,

    "TRAIN": {
        "lr": 1e-4,
        "batch_size": 4096,
        "weight_decay": 0.05,  # 进一步加大正则化，防止过拟合
    },

    "MODEL": {
        "hidden_dim": 128,  # 再次减小模型 (256 -> 128)，LSTM 参数比 GRU 多，需要减宽
        "num_layers": 2,
        "dropout": 0.5
    },

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

CONFIG["DATA_DIR"].mkdir(parents=True, exist_ok=True)
CONFIG["TEMP_DIR"].mkdir(parents=True, exist_ok=True)
CONFIG["CHECKPOINT_DIR"].mkdir(parents=True, exist_ok=True)
DEVICE = torch.device(CONFIG["DEVICE"])

ROLE_MAP = {"Grid_Bot": "网格", "Trend_Surfer": "趋势", "Scalper": "高频", "Degen": "激进", "Spot_Hodler": "囤币",
            "Bear_Raider": "空军"}
ALL_ROLES = list(ROLE_MAP.keys())

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Margaret", "Anthony", "Betty", "Donald", "Sandra", "Mark", "Ashley",
    "Paul", "Dorothy", "Steven", "Kimberly", "Andrew", "Emily", "Kenneth", "Donna",
    "George", "Michelle", "Joshua", "Carol", "Kevin", "Amanda", "Brian", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee",
    "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright", "Lopez",
    "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter",
    "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans"
]

def generate_unique_names(count):
    """生成 count 个不重复的姓名 (Name Surname_G1)"""
    names = set()
    while len(names) < count:
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        names.add(f"{fn} {ln}")
    return [f"{n}_G1" for n in names]

# 初始化名字库
ALL_UNIQUE_NAMES = generate_unique_names(CONFIG["AGENT_COUNT"])

# ==================================================================================
# 数据与模型层
# ==================================================================================
class FastNumpyDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): 
        x = torch.from_numpy(self.x[i])
        return x + torch.randn_like(x)*0.01, torch.tensor(self.y[i])


def prepare_features_static(c_hist, b_hist):
    # 如果数据长度不够，返回零矩阵
    if len(c_hist) < CONFIG["SEQ_LEN"]:
        return np.zeros((CONFIG["SEQ_LEN"], 72), dtype=np.float32)

    try:
        # 1. 原始数据提取
        closes = np.array([d['c'] for d in c_hist], dtype=np.float32)
        highs = np.array([d['h'] for d in c_hist], dtype=np.float32)
        lows = np.array([d['l'] for d in c_hist], dtype=np.float32)
        vols = np.array([d['v'] for d in c_hist], dtype=np.float32)

        # 2. EMA 平滑
        def get_ema(arr, span):
            alpha = 2 / (span + 1)
            res = np.zeros_like(arr)
            res[0] = arr[0]
            for i in range(1, len(arr)):
                res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
            return res

        smooth_c = get_ema(closes, 5)

        # 3. 特征构造 (增加 eps 防止除以零)
        eps = 1e-8

        # A. 相对位置 (Z-Score)
        mean = np.mean(smooth_c)
        std = np.std(smooth_c) + eps
        z_score = (smooth_c - mean) / std

        # B. 对数收益率
        lret = np.diff(np.log(np.maximum(smooth_c, eps)))
        lret = np.insert(lret, 0, 0) * 100

        # C. ATR
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        atr = get_ema(tr, 14)
        norm_atr = atr / (smooth_c + eps) * 1000

        # D. MACD
        ema12 = get_ema(closes, 12)
        ema26 = get_ema(closes, 26)
        dif = ema12 - ema26
        dea = get_ema(dif, 9)
        macd = (dif - dea) * 2
        norm_macd = macd / (std + eps)

        # E. KDJ-J
        rsv = (closes - lows) / (highs - lows + eps) * 100
        k = get_ema(rsv, 3)
        d = get_ema(k, 3)
        j = 3 * k - 2 * d
        norm_j = (j - 50) / 50

        # 4. 堆叠
        feats = np.stack([z_score, lret, norm_atr, norm_macd, norm_j], axis=1)

        # 5. Padding
        repeats = 72 // 5 + 1
        tiled = np.tile(feats, (1, repeats))
        padded = tiled[:, :72]

        # === 【关键修复】: 强制清洗数据，防止 GPU 崩溃 ===
        # 将 NaN (非数字) 和 Inf (无穷大) 全部替换为 0
        return np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    except Exception as e:
        # 如果计算出错，返回零矩阵，防止程序闪退
        print(f"Feature error: {e}")
        return np.zeros((len(c_hist), 72), dtype=np.float32)


class UnifiedDownloader:
    def __init__(self, target_days=15):
        self.target_days = target_days
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE

        # 确保临时目录存在
        self.temp_dir = Path("./temp_realtime_data")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self, symbols, data_dir, status_callback):
        """[历史数据] 下载到主数据目录"""
        end_ts = int(time.time() * 1000)
        # 注意：这里根据 interval 动态调整下载量，15m数据如果不下长一点，训练样本会不够
        # 如果是 15m 线，建议多下一点时间，比如 60天
        start_ts = end_ts - (self.target_days * 24 * 3600 * 1000)

        status_callback(f"📅 下载范围: {datetime.fromtimestamp(start_ts / 1000)} 至 Now")

        # 这里的 max_workers 可以根据你电脑性能调整
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda s: self._worker(s, start_ts, end_ts, data_dir, status_callback), symbols))

    def _worker(self, sym, start_ts, end_ts, data_dir, callback):
        fpath = data_dir / f"{sym}.pkl"
        data = []

        # 1. 增量更新逻辑：先读本地
        if fpath.exists():
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                if data:
                    # 如果本地有数据，从最后一条的下一刻开始下
                    start_ts = data[-1]['t'] + 1
            except:
                data = []

        # 2. 如果已经是最新的，直接返回
        if start_ts >= end_ts:
            callback(f"📥 {sym}: 无需更新")
            return data

        # === 【修复报错的核心位置】 ===
        # 必须在这里初始化 curr，这一行之前缺失或位置不对导致了 UnboundLocalError
        curr = start_ts

        src_idx = 0
        # 获取配置中的 K线 周期，默认 15m
        interval = CONFIG.get("INTERVAL", "15m")

        while curr < end_ts:
            url = f"{CONFIG['API_SOURCES'][src_idx % len(CONFIG['API_SOURCES'])]}?symbol={sym.upper()}&interval={interval}&startTime={curr}&limit=1000"
            try:
                with urllib.request.urlopen(url, context=self.ctx, timeout=5) as res:
                    batch = json.loads(res.read())
                    if not batch:
                        break  # 没有数据了

                    # 解析数据
                    new_data = [{'t': x[0], 'c': float(x[4]), 'v': float(x[5]), 'h': float(x[2]), 'l': float(x[3])} for
                                x in batch]
                    data.extend(new_data)

                    # 更新 curr 指针
                    curr = data[-1]['t'] + 1
            except Exception as e:
                # 换源重试
                src_idx += 1
                time.sleep(1)
                # 如果重试太多次还没通，跳出避免死循环
                if src_idx > 10:
                    print(f"Error downloading {sym}: {e}")
                    break

        # 保存数据
        try:
            with open(fpath, 'wb') as f:
                pickle.dump(data, f)
            callback(f"📥 {sym}: Ready ({len(data)} bars)")
        except Exception as e:
            print(f"Error saving {sym}: {e}")

        return data

    def download_temp(self, symbol, limit_count=100):
        """[实时微调] 下载到临时目录"""
        # 注意：这里的 limit_count 不再是分钟数，而是 K线根数
        interval = CONFIG.get("INTERVAL", "15m")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit_count}"
        try:
            with urllib.request.urlopen(url, context=self.ctx, timeout=5) as res:
                batch = json.loads(res.read())
                if not batch: return []
                data = [{'t': x[0], 'c': float(x[4]), 'v': float(x[5]), 'h': float(x[2]), 'l': float(x[3])} for x in
                        batch]

                fpath = self.temp_dir / f"{symbol}_temp.pkl"
                with open(fpath, 'wb') as f:
                    pickle.dump(data, f)

                return data
        except:
            return []

    # 快捷别名
    def download_recent(self, symbol, count=100):
        return self.download_temp(symbol, count)

class EvoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 替换为 LSTM
        self.lstm = nn.LSTM(
            input_size=72,
            hidden_size=CONFIG["MODEL"]["hidden_dim"],
            num_layers=CONFIG["MODEL"]["num_layers"],
            batch_first=True,
            dropout=CONFIG["MODEL"]["dropout"]
        )
        self.head = nn.Sequential(
            nn.Linear(CONFIG["MODEL"]["hidden_dim"], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        # LSTM 返回 (output, (h_n, c_n))
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步
        return self.head(out[:, -1, :])


class TechEngine:
    def __init__(self):
        self.model = EvoNet().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG["TRAIN"]["lr"], weight_decay=CONFIG["TRAIN"]["weight_decay"])
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.load_latest_checkpoint()

    def reset_optimizer(self, steps):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG["TRAIN"]["lr"], weight_decay=CONFIG["TRAIN"]["weight_decay"])
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, total_steps=steps, pct_start=0.3)

    def load_latest_checkpoint(self):
        fs = sorted(glob.glob(str(CONFIG["CHECKPOINT_DIR"]/"model_*.pth")))
        if fs: self.model.load_state_dict(torch.load(fs[-1], map_location=DEVICE))
    
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), CONFIG["CHECKPOINT_DIR"]/f"model_{datetime.now():%Y%m%d_%H%M%S}.pth")

    def run_epoch(self, loader):
        self.model.train()
        tl, st = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.unsqueeze(1).to(DEVICE, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(): loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer); self.scaler.update()
            tl += loss.item(); st += 1; self.scheduler.step()
        return tl/max(1, st)

    def evaluate_backtest(self, loader):
        self.model.eval()
        pnl, tr, opp = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                sig = (torch.sigmoid(self.model(x)).squeeze() > 0.75).float()
                if sig.sum()==0: continue
                h, m = (sig*y).sum().item(), (sig*(1-y)).sum().item()
                pnl += (h*0.008 - m*0.005); tr += (h+m); opp += len(x)
        if tr < 50: return 0.0
        return min((pnl / (CONFIG["DOWNLOAD_DAYS"]*0.2))*365, 3.0)

    def infer_batch(self, batch_x):
        """
        [GPU 加速核心] 批量推理
        batch_x: Numpy Array [Batch_Size, Seq_Len, Features]
        """
        self.model.eval()
        with torch.no_grad():
            # 1. 极速转 Tensor 并送入 GPU
            x = torch.from_numpy(batch_x).to(DEVICE)

            # 2. GPU 矩阵运算 (一次算完所有币种)
            # 输出: [Batch_Size, 1]
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            return probs

    def infer(self, c, b):
        x = torch.tensor(prepare_features_static(c, b), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # 模型输出的是 "当前是震荡市的概率"
            oscillation_prob = torch.sigmoid(self.model(x)).item()

            # 返回格式: [bear, neutral, bull]
            # 我们把 oscillation_prob 放在中间 (neutral)
            # 剩下的概率平分给 bear 和 bull (因为方向由均线决定，AI只管是否震荡)
            remain = 1 - oscillation_prob
            return [remain / 2, oscillation_prob, remain / 2], 0

# ==================================================================================
# 3. 业务层
# ==================================================================================
class Agent:
    def __init__(self, idx, name, role, genes=None, gen=1, parent="System"):
        self.id, self.name, self.role, self.generation, self.parent = idx, name, role, gen, parent
        self.balance = CONFIG["INIT_CAPITAL"]
        self.init_balance = CONFIG["INIT_CAPITAL"]
        self.positions = {}

        self.pnl = 0.0
        self.total_profit_contribution = 0.0

        if genes:
            self.genes = genes
        else:
            self.genes = {
                'tp': random.uniform(0.01, 0.04),
                'sl': random.uniform(-0.01, -0.05),
                'lev': random.randint(1, 5),
                'thresh': random.uniform(0.6, 0.85),
                'size': random.uniform(0.2, 0.9),
                'max_batch': random.randint(1, 4),
                'add_gap': random.uniform(0.005, 0.02)
            }
            if role == "Degen":
                self.genes['lev'] = random.randint(3, 5)
                self.genes['sl'] = -0.10
                self.genes['max_batch'] = random.randint(1, 2)

        self.alive = True
        self.last_trade = time.time()

        # === 新增：动态寿命 ===
        self.current_lifespan = 60  # 初始寿命 30秒

    def get_equity(self, prices):
        """
        修正后的权益计算：
        权益 = 现金余额 (Balance) + 持仓保证金 (Margin) + 浮动盈亏 (Unrealized PnL)
        """
        equity = self.balance
        for s, p in self.positions.items():
            if s in prices:
                current_price = prices[s]
                # 浮动盈亏 = (现价 - 入场价) * 数量
                unrealized_pnl = (current_price - p['entry']) * p['amt']
                # 加回保证金 (因为开仓时从 balance 扣掉了)
                equity += p['margin'] + unrealized_pnl
        return equity

    def decide(self, sym, price, probs, vol):
        if not self.alive: return

        # 1. 获取历史数据
        hist = manager.history[sym]
        # 刚启动数据不足时，放宽限制，允许只有 30根 K线就开始计算
        if len(hist) < 30: return

        # 取最近数据计算均线
        closes = [d['c'] for d in list(hist)[-50:]]

        # 简单均线策略 (MA7 vs MA25)
        # 注意：如果数据不够 25，这里会报错，所以上面加了 len check
        ma_fast = sum(closes[-7:]) / 7
        ma_slow = sum(closes[-25:]) / 25

        # 趋势判断
        trend_bull = ma_fast > ma_slow and price > ma_fast  # 价格站稳快线
        trend_bear = ma_fast < ma_slow and price < ma_fast  # 价格跌破快线

        # 2. 持仓处理 (保持趋势跟踪逻辑)
        if sym in self.positions:
            pos = self.positions[sym]
            direction = pos.get('side', 1)

            self._check(sym, price)
            if sym not in self.positions: return

            # 趋势反转平仓 (稍微放宽一点，别一抖动就平仓)
            # 只有当 价格 真的穿透 慢线(MA25) 时才反转平仓，拿住趋势
            if direction == 1 and price < ma_slow:
                self.close(sym, price)
            elif direction == -1 and price > ma_slow:
                self.close(sym, price)
            return

        # 3. 开仓决策 (空仓时)
        if self.balance < 10: return

        # === 修改 4: 放宽 AI 过滤 ===
        # probs[1] 是 AI 认为的“震荡概率”
        # 之前设 0.6 可能太严了，导致一直不开单。改为 0.8，或者刚开始训练时直接忽略
        ai_is_choppy = probs[1] > 0.7

        # 如果 AI 非常确定是震荡市，才不交易。否则哪怕 AI 没信心，只要均线出了趋势也敢冲
        if ai_is_choppy: return

        # === 修改 5: 移除出生冷却限制 ===
        # 我们在 _open 里处理冷却，这里只管信号

        if trend_bull:
            self._open(sym, price, 1)
        elif trend_bear:
            self._open(sym, price, -1)

    def _apply_slippage(self, price, qty, is_buy):
        """[V79 优化版] 小资金友好型滑点"""
        base_slippage = CONFIG["SLIPPAGE"]  # 基础滑点 万2

        volume = price * qty

        # 只有当交易额超过 10,000 U 时，才开始增加额外滑点
        # 之前是只要有量就加，对小资金太苛刻
        if volume > 10000:
            impact = ((volume - 10000) / 10000) * 0.0001
        else:
            impact = 0

        total_slip = min(0.05, base_slippage + impact)  # 上限锁死 5%
        return price * (1 + total_slip) if is_buy else price * (1 - total_slip)

    def _open(self, s, p, direction, is_add=False):
        if time.time() - self.last_trade < 15: return
        """开仓 或 加仓"""
        # 资金检查
        max_bet = 2000.0
        invest_amount = min(self.balance, max_bet) * self.genes['size']

        notional_value = invest_amount * self.genes['lev']
        fee = notional_value * CONFIG["TRANSACTION_FEE"]

        total_cost = invest_amount + fee
        if self.balance < total_cost: return
        # === 核心修改：预期收益检查 ===
        # 估算成本比例: (手续费 + 滑点) / 本金
        cost_ratio = (CONFIG["TRANSACTION_FEE"] + CONFIG["SLIPPAGE"]) * self.genes['lev']

        # 如果基因里的止盈目标 (tp) 甚至覆盖不了成本的 2倍，直接拒绝开单
        # 这会淘汰那些“短视”的交易员
        if self.genes['tp'] < cost_ratio * 2:
            return
        self.balance -= total_cost

        # 滑点
        raw_qty = notional_value / p
        is_buy = (direction == 1)
        real_entry_price = self._apply_slippage(p, raw_qty, is_buy)
        real_qty = notional_value / real_entry_price

        now = time.time()

        if is_add and s in self.positions:
            # === 加仓逻辑：合并持仓，摊薄成本 ===
            pos = self.positions[s]
            old_amt = pos['amt']
            old_margin = pos['margin']
            old_entry = pos['entry']

            # 新总数量
            new_total_amt = old_amt + real_qty
            # 新总本金
            new_total_margin = old_margin + invest_amount
            # 新加权均价 = (旧值 + 新值) / 总量
            new_avg_price = (old_entry * old_amt + real_entry_price * real_qty) / new_total_amt

            pos['amt'] = new_total_amt
            pos['entry'] = new_avg_price
            pos['margin'] = new_total_margin
            pos['batch_count'] = pos.get('batch_count', 1) + 1
            pos['last_add_price'] = real_entry_price  # 记录本次加仓价，作为下次间隔基准

            self.last_trade = time.time()  # 续命成功！

            side_str = "加多" if direction == 1 else "加空"
            manager.event_q.put(("OPEN", f"{self.name} {side_str} {s} (第{pos['batch_count']}次)"))

        else:
            # === 首次开仓 ===
            self.positions[s] = {
                'amt': real_qty,
                'entry': real_entry_price,
                'lev': self.genes['lev'],
                'margin': invest_amount,
                'side': direction,
                'open_time': now,
                'batch_count': 1,  # 初始批次
                'last_add_price': real_entry_price  # 初始基准价
            }
            self.last_trade = now
            side_str = "开多" if direction == 1 else "开空"
            manager.event_q.put(("OPEN", f"{self.name} {side_str} {s} (首仓)"))

    def _check(self, s, p):
        pos = self.positions[s]
        direction = pos.get('side', 1)

        # 计算盈亏比
        pct = (p - pos['entry']) / pos['entry'] * pos['lev'] * direction

        # 1. 硬止损 (Hard Stop)：亏损超过 25% 强制平仓，防止穿仓
        # 即使基因里写着“扛单到死”，系统也不允许
        if pct < -0.25:
            self.close(s, p)
            return

        # 1. 正常止盈止损
        if pct > self.genes['tp'] or pct < self.genes['sl']:
            self.close(s, p)
            return

        # 2. (可选) 时间止盈：持仓太久(比如2分钟)如果不亏也平了，换车
        duration = time.time() - pos['open_time']
        if duration > 120 and pct > 0.005:
             self.close(s, p)

    def close(self, s, p):
        if s not in self.positions: return 0.0
        pos = self.positions[s]

        # 1. 价格计算
        direction = pos.get('side', 1)
        # 平仓滑点：多单卖出(is_buy=False)，空单买入(is_buy=True)
        is_buy_close = (direction == -1)
        real_exit_price = self._apply_slippage(p, pos['amt'], is_buy_close)

        # 2. 原始盈亏 (Raw PnL)
        # 做多: (Exit - Entry) * Amt
        # 做空: (Entry - Exit) * Amt
        raw_pnl = (real_exit_price - pos['entry']) * pos['amt'] * direction

        # 3. 成本扣除
        margin = pos['margin']
        duration = (time.time() - pos['open_time']) / 3600
        # 借贷额
        borrowed = (pos['amt'] * pos['entry']) - margin
        # 利息 (最低按1分钟算，防止秒单无成本)
        interest = borrowed * CONFIG["LEVERAGE_INTEREST"] * max(0.016, duration)
        # 手续费
        fee = (pos['amt'] * real_exit_price) * CONFIG["TRANSACTION_FEE"]

        # 4. 净盈亏
        net_pnl = raw_pnl - interest - fee

        # === 核心修改：防止穿仓 (最大亏损不能超过本金) ===
        if net_pnl < -margin:
            net_pnl = -margin  # 亏光为止，不再倒贴

        # 5. 结算
        money_back = margin + net_pnl
        contrib = 0.0

        if net_pnl > 0:
            contrib = net_pnl * 0.70;
            keep = net_pnl * 0.30
            self.balance += margin + keep
            self.pnl += keep
            self.total_profit_contribution += contrib
            manager.event_q.put(("PROFIT", f"{self.name} 止盈 ${contrib:.2f}"))
        else:
            self.balance += money_back
            self.pnl += net_pnl

        if self.balance < 0.01: self.balance = 0  # 清理浮点数残留

        del self.positions[s]
        self.last_trade = time.time()
        self.current_lifespan += 30
        if self.current_lifespan > 180:
            self.current_lifespan = 180

        return contrib

    def check_death(self, prices):
        eq = self.get_equity(prices)

        # 1. 破产判定 (<30 U)
        if eq < 30:
            return "破产"

        # 2. 不活跃判定 (动态寿命)
        # 比较：当前空闲时间 vs 当前拥有的寿命上限
        idle_time = time.time() - self.last_trade

        if idle_time > self.current_lifespan:
            return f"不活跃 (闲置 {int(idle_time)}s > {self.current_lifespan}s)"

        return None


class Manager:
    def __init__(self):
        self.engine = TechEngine()

        self.agents = []
        # 初始化名字库逻辑
        if 'ALL_UNIQUE_NAMES' not in globals() or len(ALL_UNIQUE_NAMES) < CONFIG["AGENT_COUNT"]:
            temp_names = [f"Agent_{i}" for i in range(CONFIG["AGENT_COUNT"])]
            for i in range(CONFIG["AGENT_COUNT"]): self.agents.append(Agent(i, temp_names[i], random.choice(ALL_ROLES)))
        else:
            for i in range(CONFIG["AGENT_COUNT"]): self.agents.append(
                Agent(i, ALL_UNIQUE_NAMES[i], random.choice(ALL_ROLES)))

        self.history = defaultdict(lambda: deque(maxlen=30000))
        self.btc_history = deque(maxlen=30000)
        self.prices = {}
        self.event_q = queue.Queue()
        self.log_q = queue.Queue()
        self.group_cash = 0.0
        self.downloader = UnifiedDownloader(CONFIG["DOWNLOAD_DAYS"])

        self.status = "初始化"
        self.is_warming_up = True
        self.is_training = False

        # === 核心修改：级联熔断阈值 ===
        self.is_circuit_break = False
        self.next_meltdown_threshold = -0.10  # 初始熔断线 -10%

        init_aum = CONFIG["AGENT_COUNT"] * CONFIG["INIT_CAPITAL"]
        self.snapshot = {"cash": 0.0, "aum": init_aum, "roi": 0.0, "apy": 0.0}
        self.start_time = time.time()

        self.finance_log_file = CONFIG["FINANCE_LOG"]
        self.train_log_file = "training_metrics.csv"
        self._init_csv(self.finance_log_file, ["Time", "Group_Cash", "Equity", "AUM", "ROI", "APY"])
        self._init_csv(self.train_log_file, ["Time", "Phase", "Epoch", "Loss"])

        threading.Thread(target=self.run, daemon=True).start()

    def _init_csv(self, fpath, headers):
        if not Path(fpath).exists():
            try:
                with open(fpath, 'w', newline='') as f:
                    csv.writer(f).writerow(headers)
            except:
                pass

    def log(self, m): self.log_q.put(m); print(f"SYS: {m}")

    def load_mem(self):
        for s in CONFIG["TARGET_COINS"]:
            try:
                with open(CONFIG["DATA_DIR"]/f"{s}.pkl", 'rb') as f:
                    d = pickle.load(f)
                    self.history[s]=deque(d, maxlen=30000)
                    if s=='btcusdt': self.btc_history=deque(d, maxlen=30000)
            except: pass

    def log(self, m):
        self.log_q.put(m); print(f"SYS: {m}")

    def load_mem(self):  # 保持原样
        for s in CONFIG["TARGET_COINS"]:
            try:
                with open(CONFIG["DATA_DIR"] / f"{s}.pkl", 'rb') as f:
                    d = pickle.load(f)
                    self.history[s] = deque(d, maxlen=30000)
                    if s == 'btcusdt': self.btc_history = deque(d, maxlen=30000)
            except:
                pass

    def pretrain(self):
        """[修改版] 智能预训练：带数据安全检查"""
        self.log("正在构建训练数据...")

        # 1. 准备任务
        # 注意：这里把 1000 改小一点，刚下载数据可能还没那么多，设为 SEQ_LEN + 100 即可
        min_len = CONFIG["SEQ_LEN"] + 100
        tasks = [(s, list(self.history[s]), list(self.btc_history)) for s in CONFIG["TARGET_COINS"] if
                 len(self.history[s]) > min_len]

        if not tasks:
            self.log("⚠️ 数据不足 (正在下载中)，跳过本次训练")
            return

        # 2. 并行处理
        # 修改 proc 函数中的 Label 生成逻辑
        def proc(args):
            sym, c, b = args
            cl = np.array([x['c'] for x in c], dtype=np.float32)

            # 展望未来 6 根 K线 (1.5小时)
            future_window = 6
            if len(cl) < future_window + CONFIG["SEQ_LEN"]: return None

            # 计算未来一段时间的“绝对波动幅度”
            # 我们不关心涨还是跌，只关心波动够不够大以覆盖手续费
            future_change = (np.roll(cl, -future_window) - cl) / cl
            abs_change = np.abs(future_change)

            # 阈值：如果未来波动 > 2%，则是“值得交易的行情”(Label=0)，否则是“垃圾震荡时间”(Label=1)
            # 注意：这里 Label=0 代表 Bull/Bear (有趋势)，Label=1 代表 Neutral (震荡)
            # 我们让 AI 学习识别“垃圾时间”

            valid_indices = range(CONFIG["SEQ_LEN"], len(cl) - future_window)

            x_data = []
            y_data = []

            for i in valid_indices:
                # 如果波动大于 1.5%，标记为 0 (非震荡，适合交易)
                # 如果波动很小，标记为 1 (震荡，最好休息)
                label = 0.0 if abs_change[i] > 0.015 else 1.0

                # 只有当这是震荡市(1.0)，我们才希望 AI 输出高概率的 probs[1]
                x_data.append(prepare_features_static(c[i - CONFIG["SEQ_LEN"]:i],
                                                      b[min(len(b), i) - CONFIG["SEQ_LEN"]:min(len(b), i)]))
                y_data.append(label)

            return x_data, y_data

        self.log(f"启动 {os.cpu_count()} 核处理 {len(tasks)} 个币种...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            res = list(ex.map(proc, tasks))

        # 过滤 None
        res = [r for r in res if r is not None]

        if not res:
            self.log("❌ 有效样本为 0 (可能是市场波动太小)，跳过训练")
            return

        # 合并数据
        try:
            ax = np.concatenate([np.array(r[0], dtype=np.float32) for r in res])
            ay = np.concatenate([np.array(r[1], dtype=np.float32) for r in res])
        except Exception as e:
            self.log(f"❌ 数据合并失败: {e}")
            return

        # === 【关键检查】 ===
        if len(ax) == 0 or len(ay) == 0:
            self.log("❌ 样本集为空，无法训练")
            return

        self.log(f"构建完成: {len(ax)} 个样本")

        # 3. 开始训练
        try:
            ds = FastNumpyDataset(ax, ay)
            dl = DataLoader(ds, batch_size=1024, shuffle=True, pin_memory=False,
                            num_workers=0)  # num_workers=0 防止 Windows 崩溃
            vdl = DataLoader(ds, batch_size=1024, shuffle=False, pin_memory=False, num_workers=0)

            # 先评估
            self.log("正在评估...")
            initial_apy = self.engine.evaluate_backtest(vdl)
            self.log(f"当前模型评分: {initial_apy * 100:.1f}")

            if initial_apy > CONFIG["MIN_APY_FOR_LIVE"]:
                self.log("✅ 模型合格，直接使用")
                return

            self.log("开始强化训练...")
            self.engine.reset_optimizer(len(dl) * 10)  # 10 Epochs

            for ep in range(1, 11):
                loss = self.engine.run_epoch(dl)
                # 每 2 轮评估一次，节省时间
                if ep % 2 == 0:
                    score = self.engine.evaluate_backtest(vdl)
                    self.log(f"Ep{ep}: Loss={loss:.4f} Score={score:.2f}")

            self.engine.save_checkpoint()
            self.log("✅ 训练完成，模型已保存")

            # 清理内存
            del ax, ay, ds, dl, vdl
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            self.log(f"❌ 训练过程崩溃: {e}")
            import traceback
            traceback.print_exc()

    def quick_finetune(self):
        """[V79 修复版] 强制下载最近30分钟 + 死循环保护"""
        self.log("🔒 启动门禁：正在下载最近 30 分钟实盘数据...")

        # 1. 强制补全数据 (只补主流币，够用就行)
        core_coins = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt', 'dogeusdt']

        # 下载 45 分钟，确保 SEQ_LEN(60) 有足够的历史数据做上下文
        # 注意：这里的 45 分钟是指最新的，我们需要拼接在 history 后面
        # 但如果 history 也是刚下的，可能会重复。这里做去重合并。

        for s in core_coins:
            new_data = self.downloader.download_temp(s, 45)
            if new_data:
                # 智能合并
                if s not in self.history: self.history[s] = deque(maxlen=30000)

                current_ids = {x['t'] for x in self.history[s]}
                added_count = 0
                for d in new_data:
                    if d['t'] not in current_ids:
                        self.history[s].append(d)
                        added_count += 1
                        # 同时更新价格，防止 UI 显示 0
                        self.prices[s] = d['c']

                if s == 'btcusdt':
                    for d in new_data:
                        # btc_history 也要同步
                        if not any(x['t'] == d['t'] for x in self.btc_history):
                            self.btc_history.append(d)

                # self.log(f"  - {s}: 更新 {added_count} 条新K线")

        self.log("✅ 数据同步完成，开始校准训练...")

        # 2. 训练循环 (带最大重试次数)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                all_x, all_y = [], []

                # 遍历所有已下载数据的币种
                for s in core_coins:
                    if len(self.history[s]) < CONFIG["SEQ_LEN"] + 5: continue

                    # 取最近 100 根 (保证有足够的上下文)
                    c_data = list(self.history[s])[-100:]
                    # BTC 数据可能还没对齐，取最后的
                    b_data = list(self.btc_history)[-100:]

                    if len(b_data) < CONFIG["SEQ_LEN"]: continue

                    cl = np.array([x['c'] for x in c_data], dtype=np.float32)
                    ret = (np.roll(cl, -5) - cl) / (cl + 1e-8)

                    # 只要能切片，就生成样本
                    valid_start = CONFIG["SEQ_LEN"]
                    valid_end = len(cl) - 5

                    if valid_start >= valid_end: continue

                    for i in range(valid_start, valid_end):
                        cw = c_data[i - CONFIG["SEQ_LEN"]:i]
                        bw = b_data[min(len(b_data), i) - CONFIG["SEQ_LEN"]:min(len(b_data), i)]

                        all_x.append(prepare_features_static(cw, bw))

                        # 0.3% 门槛
                        label = 1.0 if ret[i] > 0.003 else 0.0
                        all_y.append(label)

                if not all_x:
                    self.log("⚠️ 样本依然不足 (可能刚开盘或数据未就绪)")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        break  # 放弃治疗，直接上线

                # 训练
                ds = FastNumpyDataset(np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.float32))
                dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

                self.engine.model.train()
                # 稍微加大学习率快速适应
                self.engine.optimizer.param_groups[0]['lr'] = 5e-5

                total_loss = 0
                # 跑 5 个 Epoch
                for _ in range(5):
                    for x, y in dl:
                        x, y = x.to(DEVICE), y.unsqueeze(1).to(DEVICE)
                        self.engine.optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            loss = self.engine.criterion(self.engine.model(x), y)
                        self.engine.scaler.scale(loss).backward()
                        self.engine.scaler.step(self.engine.optimizer)
                        self.engine.scaler.update()
                        total_loss += loss.item()

                self.log(f"校准完成 (Loss: {total_loss / len(dl) / 5:.4f})")
                self.engine.save_checkpoint()
                break  # 成功，退出循环

            except Exception as e:
                self.log(f"校准异常: {e}")
                break

        self.log("🚀 预热结束，交易系统正式接管")

    def inference_loop(self):
        """
        [V74 平滑版] 推理线程
        在特征提取阶段加入微小休眠，平滑 CPU 曲线。
        """
        warmup = 10
        while True:
            time.sleep(0.5)

            if self.status != "实盘交易中" or getattr(self, 'is_training', False):
                continue

            if warmup > 0:
                warmup -= 1
                continue

            batch_x = []
            valid_symbols = []
            current_prices = {}

            try:
                if len(self.btc_history) < CONFIG["SEQ_LEN"]: continue
                btc_snapshot = list(self.btc_history)[-CONFIG["SEQ_LEN"]:]

                # 遍历所有币种
                for s in CONFIG["TARGET_COINS"]:
                    if len(self.history[s]) > CONFIG["SEQ_LEN"]:
                        c_snapshot = list(self.history[s])[-CONFIG["SEQ_LEN"]:]

                        # 特征计算 (CPU密集)
                        feat = prepare_features_static(c_snapshot, btc_snapshot)

                        batch_x.append(feat)
                        valid_symbols.append(s)
                        current_prices[s] = c_snapshot[-1]['c']

                        # === 新增：每处理完一个币，休息 2ms ===
                        # 20个币总共增加 40ms 延迟，对交易无影响，但能平滑 CPU
                        time.sleep(0.002)

                if not batch_x: continue
                np_batch = np.array(batch_x, dtype=np.float32)

            except Exception:
                continue

            # GPU 推理
            try:
                probs = self.engine.infer_batch(np_batch)
            except:
                continue

            # 分发信号
            active_agents = [a for a in self.agents if a.alive]

            for i, sym in enumerate(valid_symbols):
                p_bull = probs[i]
                price = current_prices[sym]
                agent_probs = [0.1, 1 - p_bull - 0.1, p_bull]

                if p_bull > 0.6:
                    for a in active_agents:
                        if sym in a.positions:
                            c = a.close(sym, price)
                            self.group_cash += c
                        a.decide(sym, price, agent_probs, 0)
                else:
                    for a in active_agents:
                        if sym in a.positions:
                            c = a.close(sym, price)
                            self.group_cash += c

    def ecosystem_loop(self):
        """
        [V80 进化版]
        1. 严格优胜劣汰，资金池不再无限输血
        2. 5% 物种多样性保护
        """
        while True:
            time.sleep(1)
            if self.status != "实盘交易中": continue

            dead_idxs = []

            # 统计当前各流派人数
            role_cnt = Counter([a.role for a in self.agents])
            # 5% 保底线 (500 * 0.05 = 25人)
            MIN_SPECIES_COUNT = int(CONFIG["AGENT_COUNT"] * 0.05)

            # 1. 死亡判定与结算
            for i, a in enumerate(self.agents):
                cause = a.check_death(self.prices)

                # 增加一个条件：如果连续亏损导致余额低于 30U (本金100)，直接强制止损淘汰
                # 避免像之前那样亏到 0 还在跑
                if a.balance < 30:
                    cause = "破产清算"

                if cause:
                    # 记录财务变动
                    # 只有当余额大于0时，才算回收了残值
                    scrap_value = max(0, a.balance)
                    self.group_cash += scrap_value

                    self.event_q.put(("DEATH", f"{a.name} ({a.role}) {cause} 离场 (回收 ${scrap_value:.1f})"))
                    dead_idxs.append(i)
                    role_cnt[a.role] -= 1

            # 2. 繁殖与重生 (优胜劣汰核心)
            if dead_idxs:
                # 选出精英父母 (赚钱且存活的)
                # 按照 (总权益 + 已上缴利润) 排序
                parents = sorted(
                    [a for a in self.agents if a.alive and (a.balance + a.total_profit_contribution > 105)],
                    key=lambda x: x.balance + x.total_profit_contribution,
                    reverse=True
                )
                # 取前 20% 作为种马
                top_tier = parents[:max(1, int(len(parents) * 0.2))]

                for i in dead_idxs:
                    # 新人入场，集团必须掏 100U 成本
                    # 如果集团没钱了 (Group Cash < 100)，理论上游戏结束或必须借贷
                    # 这里为了演示，允许负债经营，但在 UI 上会显示惨烈的红色
                    self.group_cash -= CONFIG["INIT_CAPITAL"]

                    old_role = self.agents[i].role

                    # === 决策：是“保底重生”还是“进化迭代”？ ===

                    # 情况 A: 该流派濒临灭绝 (<5%) -> 强制补充该流派 (Mutation)
                    if role_cnt[old_role] < MIN_SPECIES_COUNT:
                        new_role = old_role
                        origin = "Species_Prot"  # 物种保护
                        # 基因完全随机重置 (因为旧的太菜了)
                        new_genes = None
                        role_cnt[old_role] += 1  # 计数回补

                        fn = random.choice(FIRST_NAMES)
                        ln = random.choice(LAST_NAMES)
                        new_name = f"{fn} {ln}_G1"
                        new_gen = 1

                    # 情况 B: 该流派人够多 -> 允许优胜劣汰 (Crossover)
                    else:
                        # 80% 概率继承精英，20% 概率随机突变 (引入新血)
                        if top_tier and random.random() < 0.8:
                            p_obj = random.choice(top_tier)
                            new_role = p_obj.role  # 继承赢家的职业！
                            new_genes = p_obj.genes.copy()

                            # 基因微调 (变异)
                            for k in new_genes:
                                # 稍微波动 5%
                                new_genes[k] *= random.uniform(0.95, 1.05)

                            # 强制基因锁：防止进化出超级高倍杠杆自杀
                            new_genes['lev'] = min(5.0, max(1.0, new_genes['lev']))

                            origin = f"Clone_{p_obj.name.split('_')[0]}"
                            core = p_obj.name.rsplit('_G', 1)[0].replace("*", "")
                            new_gen = p_obj.generation + 1
                            new_name = f"{core}_G{new_gen}"

                            role_cnt[new_role] += 1
                        else:
                            # 随机突变 (引入鲶鱼)
                            new_role = random.choice(ALL_ROLES)
                            new_genes = None
                            origin = "Random_Hire"
                            fn = random.choice(FIRST_NAMES)
                            ln = random.choice(LAST_NAMES)
                            new_name = f"{fn} {ln}_G1"
                            new_gen = 1
                            role_cnt[new_role] += 1

                    # 创建新 Agent
                    self.agents[i] = Agent(i, new_name, new_role, new_genes, new_gen, origin)
                    self.event_q.put(("BIRTH", f"{new_name} ({new_role}) 入职 [{origin}]"))

            # 3. 严格财务核算
            equity_agents = sum(a.get_equity(self.prices) for a in self.agents)
            total_aum = self.group_cash + equity_agents
            # 初始投入永远是 500 * 100 = 50000 (不随时间变化)
            total_invested = CONFIG["AGENT_COUNT"] * CONFIG["INIT_CAPITAL"]

            roi = (total_aum - total_invested) / total_invested
            days = (time.time() - self.start_time) / 86400
            apy = roi / max(days, 1e-5) * 365

            self.snapshot = {
                "cash": self.group_cash,
                "aum": total_aum,
                "roi": roi,
                "apy": apy,
                "top_agents": sorted(self.agents, key=lambda x: x.pnl, reverse=True)[:5]
            }

            # === 核心修改：级联熔断 ===
            # 如果 ROI 跌破当前阈值 (-10%, -20%...)
            if roi < self.next_meltdown_threshold and not self.is_circuit_break and not self.is_training:
                self.is_circuit_break = True
                self.next_meltdown_threshold -= 0.10  # 阈值下移到 -20%

                self.status = f"⚠️ 触发熔断 (ROI<{self.next_meltdown_threshold + 0.1:.0%})"
                self.log(f"🚨 风控警报：ROI {roi * 100:.1f}% 触及熔断线！停止交易，准备紧急校准...")

            if int(time.time()) % 60 == 0:
                try:
                    with open(self.finance_log_file, 'a', newline='') as f:
                        csv.writer(f).writerow(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{self.group_cash:.2f}", f"{equity:.2f}",
                             f"{total_aum:.2f}", f"{roi:.4f}", f"{apy:.2f}"])
                except:
                    pass

    def ws_loop(self):
        """
        [全功能版] 数据接收 + 预热检查 + 训练锁 + 交易执行
        """

        async def loop():
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            # 预热计数器 (防止刚启动时的数据断层导致乱开单)
            warmup_counter = 10

            while True:
                try:
                    url = f"{CONFIG['WSS_URL']}/stream?streams={'/'.join([s + '@kline_1m' for s in CONFIG['TARGET_COINS']])}"
                    async with websockets.connect(url, ssl=ctx) as ws:
                        self.log("🔌 交易所数据流已连接")

                        while True:
                            msg = json.loads(await ws.recv())
                            k = msg['data']['k']
                            s = msg['data']['s'].lower()

                            # 1. 核心任务：更新数据 (永不中断)
                            d = {'t': k['t'], 'c': float(k['c']), 'v': float(k['v']), 'h': float(k['h']),
                                 'l': float(k['l'])}
                            self.history[s].append(d)
                            self.prices[s] = d['c']
                            if s == 'btcusdt': self.btc_history.append(d)

                            # 2. 状态检查：如果正在预热收集数据，或者正在后台微调，则不进行交易
                            if self.is_warming_up or self.is_training or self.is_circuit_break:
                                continue

                            # 3. 预热缓冲：跳过连接后的前几条数据
                            if warmup_counter > 0:
                                if s == 'btcusdt':
                                    warmup_counter -= 1
                                    if warmup_counter == 0: self.log("🔥 预热缓冲结束，交易引擎启动")
                                continue

                            # 4. 推理与交易 (高频模式)
                            if len(self.history[s]) > CONFIG["SEQ_LEN"]:
                                # 提取数据快照
                                ch = list(self.history[s])
                                bh = list(self.btc_history)

                                # AI 推理
                                p, _ = self.engine.infer(ch, bh)

                                if p:
                                    # 分发给活着的交易员
                                    active_agents = [a for a in self.agents if a.alive]
                                    for a in active_agents:
                                        # 平仓检测 (含止盈止损)
                                        if s in a.positions:
                                            c = a.close(s, d['c'])
                                            self.group_cash += c  # 计入集团已实现盈亏

                                        # 开仓检测
                                        a.decide(s, d['c'], p, 0)

                except Exception as e:
                    self.log(f"WS重连: {e}")
                    await asyncio.sleep(5)

        asyncio.run(loop())


    def run(self):
        if getattr(self, '_is_running', False): return
        self._is_running = True

        self.status = "下载数据..."
        self.downloader.download_all(CONFIG["TARGET_COINS"], CONFIG["DATA_DIR"], self.log)
        self.load_mem()

        self.status = "预训练..."
        self.pretrain()

        self.status = "启动线程..."
        threading.Thread(target=self.ws_loop, daemon=True).start()
        threading.Thread(target=self.ecosystem_loop, daemon=True).start()
        threading.Thread(target=self.finetune_loop, daemon=True).start()

        # === 修改点：直接进入校准，不再死等 ===
        self.status = "实盘校准..."
        self.quick_finetune()

        self.is_warming_up = False
        self.status = "实盘交易中"
        self.log("🚀 系统正式上线")

    def ecosystem_loop(self):
        """
        [V68 修正版] 严谨财务核算与进化循环
        """
        while True:
            time.sleep(1)
            if self.status != "实盘交易中": continue

            dead_idxs = []
            role_cnt = Counter([a.role for a in self.agents])
            min_prot = max(2, int(CONFIG["AGENT_COUNT"] * 0.05))

            # 1. 死亡判定
            for i, a in enumerate(self.agents):
                cause = a.check_death(self.prices)
                if cause:
                    if role_cnt[a.role] <= min_prot:
                        # 濒危保护：集团注资重置 (亏损 = 100 - balance)
                        refill_cost = CONFIG["INIT_CAPITAL"] - a.balance
                        self.group_cash -= refill_cost

                        a.balance = CONFIG["INIT_CAPITAL"]
                        a.init_balance = CONFIG["INIT_CAPITAL"]
                        a.last_trade = time.time()
                        a.name = a.name.split("*")[0] + "*"
                        for k in a.genes: a.genes[k] *= random.uniform(0.9, 1.1)
                    else:
                        # 真正死亡：回收残值，稍后新人注资会扣除 100
                        self.group_cash += a.balance
                        self.event_q.put(("DEATH", f"{a.name} 离场 (剩 ${a.balance:.1f})"))
                        dead_idxs.append(i)
                        role_cnt[a.role] -= 1

            # 2. 繁殖与重生
            if dead_idxs:
                parents = sorted([a for a in self.agents if a.pnl > 0 and a.alive], key=lambda x: x.pnl, reverse=True)
                top10 = parents[:max(1, int(len(parents) * 0.1))]

                for i in dead_idxs:
                    self.group_cash -= CONFIG["INIT_CAPITAL"]

                    r = random.random();
                    new_role = random.choice(ALL_ROLES);
                    new_genes = None;
                    p_obj = None

                    if top10 and r < 0.6:  # 60% 继承
                        p_obj = random.choice(top10)
                        new_genes = p_obj.genes.copy()
                        new_role = p_obj.role
                        new_gen = p_obj.generation + 1
                        core = p_obj.name.rsplit('_G', 1)[0].replace("*", "")
                        new_name = f"{core}_G{new_gen}"
                        origin = f"{p_obj.name}"

                        # 基因微调
                        for k in new_genes: new_genes[k] *= random.uniform(0.95, 1.05)

                        # === 核心修正：基因锁 (Gene Lock) ===
                        # 强制将杠杆限制在 5倍以内，无论怎么变异
                        new_genes['lev'] = min(5.0, max(1.0, new_genes['lev']))
                        # 强制止损不能太小
                        new_genes['sl'] = min(-0.005, new_genes['sl'])

                    else:
                        fn = random.choice(FIRST_NAMES);
                        ln = random.choice(LAST_NAMES)
                        new_name = f"{fn} {ln}_G1";
                        new_gen = 1;
                        origin = "Mutation"
                        # 新人默认基因锁也在 Agent 初始化里限制了，这里无需额外操作

                    self.agents[i] = Agent(i, new_name, new_role, new_genes, new_gen, origin)
                    self.event_q.put(("BIRTH", f"{new_name} 入职 (From {origin})"))

            # 3. 财务报表 (绝对严谨版)
            # 固定总投入 = 人数 * 100 (假设集团只有这一笔初始资金，后续都是利润留存或亏损)
            total_invested = CONFIG["AGENT_COUNT"] * CONFIG["INIT_CAPITAL"]

            # 活人手里的钱
            equity_agents = sum(a.get_equity(self.prices) for a in self.agents)

            # 总资产 (AUM) = 金库现金 + 活人权益
            # group_cash 在初始化时是 0，每次新人出生 -100 (如果是初始化时扣除则初始为-3万，这里逻辑是动态扣除)
            # 修正逻辑：为了计算 ROI，我们需要一个清晰的净值。
            # 假设初始时刻：金库=0，Agent手持3万。AUM=3万。投入=3万。ROI=0。
            # 运行后：group_cash 记录了 (回收残值 - 再投入成本 + 分红)。
            # 所以 group_cash 可以是负数（如果一直还要补贴新人）。
            total_aum = self.group_cash + equity_agents

            # 净利润
            net_profit = total_aum - total_invested

            # ROI
            roi = net_profit / total_invested

            # 年化
            days = (time.time() - self.start_time) / 86400
            apy = roi / max(days, 1e-5) * 365

            self.snapshot = {
                "cash": self.group_cash,
                "aum": total_aum,
                "roi": roi,
                "apy": apy,
                "top_agents": sorted(self.agents, key=lambda x: x.pnl, reverse=True)[:int(CONFIG["AGENT_COUNT"] * 0.1)]
            }

            if int(time.time()) % 60 == 0:
                try:
                    with open(self.finance_log_file, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{self.group_cash:.2f}",
                            f"{equity_agents:.2f}",
                            f"{total_aum:.2f}",
                            f"{roi:.4f}",
                            f"{apy:.2f}"
                        ])
                except:
                    pass

    def finetune_loop(self):
        """
        [V78 极速熔断版]
        1. 秒级响应熔断信号
        2. 盘中微调只训练最近 5-10 分钟的行为模式
        """
        # 初始等待一会，让系统先跑起来
        time.sleep(10)

        while True:
            # === 1. 智能休眠 (核心修改) ===
            # 如果没熔断，就睡 30 分钟，但每秒检查一次是否触发了熔断
            target_sleep = 1800
            for _ in range(target_sleep):
                if self.is_circuit_break:
                    self.log("🚨 监测到熔断信号，立即中断休眠，开始急救！")
                    break
                time.sleep(1)

            # 如果既不是实盘，也没熔断，就继续挂起
            if self.status != "实盘交易中" and not self.is_circuit_break:
                continue

            was_circuit_break = self.is_circuit_break
            self.is_training = True

            # === 2. 补数据 (盘前10分钟 / 盘中5分钟) ===
            # 注意：下载必须足够长以覆盖 SEQ_LEN，但我们只取最后一点做 label
            download_len = 65  # 至少要有 60 (SEQ_LEN) + 5 (Target)

            if was_circuit_break:
                self.log("🚑 [熔断] 正在获取最新行情...")
                for s in CONFIG["TARGET_COINS"]:
                    # 补齐数据
                    data = self.downloader.download_recent(s, download_len)
                    if data:
                        exist = {x['t'] for x in self.history[s]}
                        for d in data:
                            if d['t'] not in exist:
                                self.history[s].append(d)
                                if s == 'btcusdt': self.btc_history.append(d)

            mode_str = "🚑 紧急修复" if was_circuit_break else "🧠 定期微调"
            self.status = f"{mode_str}..."

            try:
                # 熔断时训练所有币，平时只抽 3 个
                target_coins = CONFIG["TARGET_COINS"] if was_circuit_break else random.sample(CONFIG["TARGET_COINS"], 3)

                all_x, all_y = [], []

                for s in target_coins:
                    if len(self.history[s]) < CONFIG["SEQ_LEN"] + 5: continue

                    # 提取最近数据
                    # 我们需要 SEQ_LEN 的历史来预测，但只关心最近几分钟的 Label
                    # 比如：取最近 70 分钟数据 -> 构造出 5-10 个样本
                    window_size = CONFIG["SEQ_LEN"] + 10

                    c_data = list(self.history[s])[-window_size:]
                    b_data = list(self.btc_history)[-window_size:]

                    if len(c_data) < window_size: continue

                    cl = np.array([x['c'] for x in c_data], dtype=np.float32)
                    ret = (np.roll(cl, -5) - cl) / (cl + 1e-8)

                    # 只训练最后 5-10 个点 (反映最近 5-10 分钟的盘口特征)
                    for i in range(CONFIG["SEQ_LEN"], len(cl) - 5):
                        thr = 0.0020

                        if ret[i] > thr:
                            label = 1.0  # 涨
                        elif ret[i] < -thr:
                            label = 0.0  # 跌
                        else:
                            # 震荡区间，不要丢弃！
                            # 标记为 0.5 (中性)，让模型学会“不确定”
                            # 但因为我们是二分类 (BCE)，0.5 会让 Sigmoid 输出 0.5，正好不上不下
                            label = 0.5

                        all_x.append(prepare_features_static(cw, bw))
                        all_y.append(label)
                    time.sleep(0.002)

                if all_x:
                    gc.collect()
                    ds = FastNumpyDataset(np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.float32))
                    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

                    self.engine.model.train()
                    # 熔断时加大学习率
                    lr = 1e-4 if was_circuit_break else 1e-5
                    self.engine.optimizer.param_groups[0]['lr'] = lr

                    # 快速跑 2 轮
                    total_loss = 0
                    for _ in range(2):
                        for x, y in dl:
                            x, y = x.to(DEVICE), y.unsqueeze(1).to(DEVICE)
                            self.engine.optimizer.zero_grad()
                            with torch.cuda.amp.autocast():
                                loss = self.engine.criterion(self.engine.model(x), y)
                            self.engine.scaler.scale(loss).backward()
                            self.engine.scaler.step(self.engine.optimizer)
                            self.engine.scaler.update()
                            total_loss += loss.item()
                            time.sleep(0.005)

                    self.log(f"{mode_str} 完成. Loss: {total_loss / len(dl) / 2:.4f}")
                    self.engine.save_checkpoint()

                    del ds, dl, x, y
                    torch.cuda.empty_cache()
                else:
                    self.log("数据波动过小，跳过本次微调")

            except Exception as e:
                self.log(f"微调异常: {e}")

            # 3. 恢复
            self.is_training = False
            if self.is_circuit_break:
                self.is_circuit_break = False
                self.log("▶️ 熔断解除，交易重启")
            self.status = "实盘交易中"



# ==================================================================================
# 4. 旗舰 UI (V71 极速响应版：按需渲染 + 防卡死)
# ==================================================================================
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("银河帝国 V75 - 旗舰作战室 (零卡顿版)")
        self.geometry("1800x1000")
        self.state('zoomed')
        self.configure(bg="#0b0b0b")

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        style = ttk.Style()
        style.theme_use("clam")
        # 背景色 #121212 (深灰)，前景色 white (白)
        style.configure("Treeview", background="#121212", foreground="white", fieldbackground="#121212", rowheight=25,
                        borderwidth=0)
        style.configure("Treeview.Heading", background="#2d2d2d", foreground="#aaa",
                        font=("Microsoft YaHei", 9, "bold"))
        style.map('Treeview', background=[('selected', '#2979ff')])

        # 顶部财报
        f_top = tk.Frame(self, bg="#1a1a1a", height=80);
        f_top.pack(fill="x", padx=5, pady=5)
        self.lbl_cash = self.mk_card(f_top, "集团金库", "$0", "#00e676")
        self.lbl_aum = self.mk_card(f_top, "总资产 (AUM)", "$0", "#2979ff")
        self.lbl_roi = self.mk_card(f_top, "ROI", "0.00%", "#ffea00")
        self.lbl_apy = self.mk_card(f_top, "APY", "0.00%", "#d500f9")
        self.lbl_status = self.mk_card(f_top, "状态", "Init", "#bdbdbd")

        # 主体
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#0b0b0b", sashwidth=4)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 左栏 ---
        f_left = tk.Frame(paned, bg="#121212");
        paned.add(f_left, width=900)

        f_head = tk.Frame(f_left, bg="#2d2d2d");
        f_head.pack(fill="x")
        tk.Label(f_head, text="全员实时监控", bg="#2d2d2d", fg="white", font=("Microsoft YaHei", 10)).pack(side="left",
                                                                                                           padx=10)

        self.page = 0;
        self.page_size = 50
        f_pager = tk.Frame(f_head, bg="#2d2d2d");
        f_pager.pack(side="right", padx=5)
        tk.Button(f_pager, text="<", command=lambda: self.ch_page(-1), bg="#444", fg="white", relief="flat").pack(
            side="left")
        self.lbl_page = tk.Label(f_pager, text="1/1", bg="#2d2d2d", fg="#aaa");
        self.lbl_page.pack(side="left", padx=5)
        tk.Button(f_pager, text=">", command=lambda: self.ch_page(1), bg="#444", fg="white", relief="flat").pack(
            side="left")

        cols = ("ID", "代", "姓名", "流派", "基因来源", "权益", "贡献", "杠杆", "状态")
        self.tv_agents = ttk.Treeview(f_left, columns=cols, show="headings", height=20)
        widths = [40, 40, 110, 60, 100, 70, 70, 40, 50]
        for c, w in zip(cols, widths): self.tv_agents.heading(c, text=c); self.tv_agents.column(c, width=w,
                                                                                                anchor="center")

        sb = ttk.Scrollbar(f_left, orient="vertical", command=self.tv_agents.yview);
        self.tv_agents.configure(yscrollcommand=sb.set)
        self.tv_agents.pack(side="left", fill="both", expand=True);
        sb.pack(side="right", fill="y")
        self.tv_agents.bind("<Double-1>", self.show_detail)
        self.tv_agents.tag_configure("dying", foreground="#ff1744");
        self.tv_agents.tag_configure("rich", foreground="#ffea00")

        # === 核心优化：预先创建占位符 ===
        # 不要每次都删除重建，而是初始化好 50 个空行，只更新它们的值
        for _ in range(self.page_size):
            self.tv_agents.insert("", "end", values=("",) * 9)

        # --- 右栏 ---
        f_right = tk.Frame(paned, bg="#121212");
        paned.add(f_right, width=800)
        self.nb = ttk.Notebook(f_right)
        self.nb.pack(fill="both", expand=True)

        # Tab 1: 矩阵
        t_matrix = tk.Frame(self.nb, bg="black");
        self.nb.add(t_matrix, text="🌌 银河矩阵")
        f_leg = tk.Frame(t_matrix, bg="black");
        f_leg.pack(fill="x", pady=5, padx=10)
        for c, t in [("#2979ff", "活跃"), ("#00e676", "盈利"), ("#d500f9", "满级"),
                     ("#ff9100", "亏损"), ("#ff1744", "濒死"), ("#333333", "静默")]:
            f = tk.Frame(f_leg, bg="black");
            f.pack(side="left", padx=6)
            tk.Label(f, text="●", fg=c, bg="black", font=("Arial", 12)).pack(side="left")
            tk.Label(f, text=t, fg="#aaa", bg="black", font=("Microsoft YaHei", 9)).pack(side="left", padx=2)
        self.cv_matrix = tk.Canvas(t_matrix, bg="#050505", highlightthickness=0)
        self.cv_matrix.pack(fill="both", expand=True, padx=10, pady=10)
        self.cv_matrix.bind("<Configure>", self.on_matrix_resize)

        # Tab 2: 图表
        t_chart = tk.Frame(self.nb, bg="#121212");
        self.nb.add(t_chart, text="📊 数据分析")
        self.fig = Figure(figsize=(5, 8), dpi=100, facecolor="#121212")
        self.ax1 = self.fig.add_subplot(311);
        self.ax2 = self.fig.add_subplot(312);
        self.ax3 = self.fig.add_subplot(313)
        self.fig.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=t_chart);
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tab 3: 日志
        t_log = tk.Frame(self.nb, bg="#121212");
        self.nb.add(t_log, text="📜 交易广场")
        self.txt_log = scrolledtext.ScrolledText(t_log, bg="#000", fg="#0f0", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)
        for t, c in [("DEATH", "#ff1744"), ("BIRTH", "#00e676"), ("PROFIT", "#ffea00")]: self.txt_log.tag_config(t,
                                                                                                                 foreground=c)

        self.matrix_nodes = []
        self.matrix_initialized = False
        self.current_tab = 0
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # 启动循环
        self.update_ui_fast()  # 100ms
        self.update_ui_slow()  # 1000ms

    def mk_card(self, p, t, v, c):
        f = tk.Frame(p, bg="#1a1a1a");
        f.pack(side="left", fill="y", expand=True, padx=10)
        tk.Label(f, text=t, bg="#1a1a1a", fg="#aaa", font=("Microsoft YaHei", 9)).pack(anchor="w")
        fg = c
        try:
            if float(str(v).replace('$', '').replace('%', '').replace(',', '')) < 0: fg = "#ff1744"
        except:
            pass
        tk.Label(f, text=v, bg="#1a1a1a", fg=fg, font=("Arial", 20, "bold")).pack(anchor="w")
        return f

    def ch_page(self, d):
        self.page += d

    def on_tab_change(self, e):
        try:
            self.current_tab = self.nb.index("current")
        except:
            pass
        if self.current_tab == 1: self.update_charts()  # 切到图表时才刷新一次

    def on_matrix_resize(self, e):
        self.matrix_initialized = False

    def init_matrix_grid(self):
        agents = manager.agents
        if not agents: return
        w = self.cv_matrix.winfo_width();
        h = self.cv_matrix.winfo_height()
        if w < 50: return
        self.cv_matrix.delete("all");
        self.matrix_nodes = []
        total = len(agents)
        ar = w / h;
        rows = max(1, int(math.sqrt(total / ar)));
        cols = int(math.ceil(total / rows))
        while rows * cols < total: cols += 1
        cw = w / cols;
        ch = h / rows;
        rad = min(cw, ch) * 0.35
        for i in range(total):
            r, c = divmod(i, cols)
            cx, cy = c * cw + cw / 2, r * ch + ch / 2
            oid = self.cv_matrix.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#333", outline="")
            self.matrix_nodes.append(oid)
        self.matrix_initialized = True

    def update_ui_fast(self):
        """[极速循环] 100ms: 矩阵 + 日志 (修复全红Bug版)"""
        # 1. 矩阵渲染
        if self.current_tab == 0:
            if not self.matrix_initialized: self.init_matrix_grid()
            if self.matrix_nodes:
                now = time.time()
                # 降低一点刷新频率，防止闪瞎眼，每秒刷新 10 次
                frame_idx = int(now * 10) % 5
                limit = min(len(manager.agents), len(self.matrix_nodes))

                for i in range(limit):
                    # 分批渲染，降低CPU压力
                    if i % 5 != frame_idx: continue

                    a = manager.agents[i]
                    eq = a.get_equity(manager.prices)
                    idle = now - a.last_trade

                    # 获取当前寿命上限 (默认60s)
                    max_life = getattr(a, 'current_lifespan', 60)
                    # 计算剩余时间
                    remaining = max_life - idle

                    # === 核心修正：基于生命百分比渲染 ===
                    # 0.0 (满血) -> 1.0 (死亡)
                    death_progress = idle / max(1, max_life)

                    # --- 第一层：基础颜色 (盈亏) ---
                    col = "#424242"  # 默认深灰 (比背景稍亮)

                    if eq < a.init_balance * 0.95:
                        col = "#ff9800"  # 亏损 (橙色，不要太红，留给濒死)
                    elif eq > a.init_balance * 1.02:
                        col = "#00e676"  # 盈利 (绿色)
                    elif eq > a.init_balance * 1.2:
                        col = "#ffea00"  # 暴富 (金色)

                    # --- 第二层：特殊状态覆盖 ---

                    # 1. 满级大佬 (紫色常驻)
                    if max_life >= 180:
                        col = "#d500f9"

                        # 2. 濒死警告 (优先级最高 - 红色闪烁)
                    # 条件：剩余时间 < 10秒 或 生命条只剩 15%
                    if remaining < 10 or death_progress > 0.85:
                        # 快速闪烁 (警报红 vs 暗红)
                        col = "#ff1744" if int(now * 4) % 2 == 0 else "#500000"

                    # 3. 刚交易完 (蓝色高亮 - 瞬间反馈)
                    # 刚开单/平单 1.5秒内显示蓝色
                    if idle < 1.5:
                        col = "#2979ff"

                    # 应用颜色
                    self.cv_matrix.itemconfig(self.matrix_nodes[i], fill=col)

        # 2. 日志刷新 (保持不变)
        new_logs = []
        while not manager.event_q.empty():
            try:
                new_logs.append(manager.event_q.get_nowait())
            except:
                break

        if new_logs:
            self.txt_log.configure(state='normal')
            for t, m in new_logs:
                self.txt_log.insert("end", f"[{datetime.now():%H:%M:%S}] {m}\n", t)

            if int(self.txt_log.index('end-1c').split('.')[0]) > 100:
                self.txt_log.delete("1.0", "2.0")

            if self.current_tab == 2: self.txt_log.see("end")

        self.after(100, self.update_ui_fast)

    def update_ui_slow(self):
        """[慢速循环] 1000ms: 列表 + 财报 + 图表"""
        # 1. 刷新财报 (保持不变)
        try:
            self.lbl_status.winfo_children()[1].config(text=manager.status)
            s = manager.snapshot
            if s:
                def fmt(v, is_pct=False):
                    if abs(v) > 1e6: return f"{v:.2e}%" if is_pct else f"${v:.2e}"
                    return f"{v * 100:.2f}%" if is_pct else f"{'-' if v < 0 else ''}${abs(v):,.0f}"

                self.lbl_cash.winfo_children()[1].config(text=fmt(s['cash']))
                self.lbl_aum.winfo_children()[1].config(text=fmt(s['aum']))
                self.lbl_roi.winfo_children()[1].config(text=fmt(s['roi'], True))
                self.lbl_apy.winfo_children()[1].config(text=fmt(s['apy'], True))
        except:
            pass

        # 2. 列表刷新 (1秒一次)
        if int(time.time()) % 1 == 0:
            for x in self.tv_agents.get_children(): self.tv_agents.delete(x)

            # 按 (剩余寿命 + 权益) 综合排序，让快死的人沉底，活得久且有钱的在上面
            now = time.time()
            sorted_ag = sorted(
                manager.agents,
                key=lambda x: (getattr(x, 'current_lifespan', 60) - (now - x.last_trade)) + (x.balance * 0.1),
                reverse=True
            )

            tp = (len(sorted_ag) + self.page_size - 1) // self.page_size
            if tp < 1: tp = 1
            if self.page >= tp: self.page = tp - 1
            if self.page < 0: self.page = 0
            self.lbl_page.config(text=f"{self.page + 1}/{tp}")

            start = self.page * self.page_size
            for i, a in enumerate(sorted_ag[start: start + self.page_size]):
                eq = a.get_equity(manager.prices)
                idle = now - a.last_trade

                # === 新增：倒计时显示 ===
                max_life = getattr(a, 'current_lifespan', 60)
                remaining = max(0, int(max_life - idle))

                # 状态字符串: "45s / 60s"
                status_str = f"{remaining}s / {max_life}s"

                # 存活时间标签 (显示总共活了多久，作为"代"的补充)
                # 这里我们复用"存活"列，原来是显示 0s前，现在可以显示 "Level X"
                # 每 30s 算 1级，满级 180s = Lv6
                level = int(max_life / 30)
                life_level_str = f"Lv.{level}"

                # 颜色 Tag
                tag = "normal"
                if remaining < 10:
                    tag = "dying"
                elif max_life >= 180:
                    tag = "legend"  # 传说级
                elif eq > a.init_balance * 1.2:
                    tag = "rich"

                # 插入行
                # 列顺序: ID, 代, 姓名, 流派, 基因来源, 权益, 贡献, 杠杆, 状态(倒计时)
                self.tv_agents.insert("", "end", values=(
                    a.id,
                    a.generation,
                    a.name,
                    ROLE_MAP.get(a.role, a.role),
                    life_level_str,  # 原 基因来源 列现在显示 等级
                    f"${eq:.1f}",
                    f"${a.total_profit_contribution:.1f}",
                    f"x{a.genes['lev']}",
                    status_str  # 原 状态 列显示倒计时
                ), tags=(tag,))

            # 配置 Tag 颜色
            self.tv_agents.tag_configure("normal", foreground="white")
            self.tv_agents.tag_configure("dying", foreground="#ff1744")  # 红色警告
            self.tv_agents.tag_configure("rich", foreground="#00e676")  # 绿色富豪
            self.tv_agents.tag_configure("legend", foreground="#d500f9")  # 紫色传说

        # 3. 日志 (保持不变)
        while not manager.event_q.empty():
            t, m = manager.event_q.get()
            if float(self.txt_log.index('end')) > 200: self.txt_log.delete('1.0', '2.0')
            self.txt_log.insert("end", f"[{datetime.now():%H:%M:%S}] {m}\n", t)
            if self.current_tab == 2: self.txt_log.see("end")

        # 4. 图表
        if int(time.time()) % 5 == 0: self.update_charts()

        self.after(1000, self.update_ui_slow)

    def update_charts(self):
        ag = manager.agents
        if not ag: return
        # 绘图逻辑比较重，放在 try 块里防止闪退
        try:
            rc = Counter([a.role for a in ag])
            self.ax1.clear()
            self.ax1.pie(rc.values(), labels=[ROLE_MAP.get(k, k) for k in rc.keys()], autopct='%1.0f%%',
                         textprops={'color': "w", 'fontsize': 8, 'fontfamily': 'Microsoft YaHei'}, startangle=90)
            self.ax1.set_title("职业分布", color="w", fontsize=9, fontfamily='Microsoft YaHei')

            gens = [a.generation for a in ag]
            if gens:
                self.ax2.clear();
                self.ax2.set_facecolor("#1e1e1e")
                min_g, max_g = min(gens), max(gens)
                if max_g - min_g <= 15:
                    gc = Counter(gens);
                    gs = sorted(gc.keys())
                    self.ax2.bar(gs, [gc[g] for g in gs], color="#2979ff", alpha=0.7);
                    self.ax2.set_xticks(gs)
                else:
                    self.ax2.hist(gens, bins=min(20, int(max_g - min_g) + 1), color="#2979ff", alpha=0.7)
                self.ax2.set_title("代际演化", color="w", fontsize=9, fontfamily='Microsoft YaHei')
                self.ax2.tick_params(colors='w', labelsize=7)

            self.ax3.clear();
            self.ax3.set_facecolor("#1e1e1e")
            rp = defaultdict(float)
            for role, pnl in manager.role_history_pnl.items(): rp[role] += pnl
            for a in ag: rp[a.role] += (a.get_equity(manager.prices) - a.init_balance + a.pnl)
            rls = list(rp.keys());
            pnls = [rp[r] for r in rls]
            if rls:
                clrs = ["#00e676" if v > 0 else "#ff1744" for v in pnls]
                xl = [ROLE_MAP.get(r, r) for r in rls]
                self.ax3.bar(xl, pnls, color=clrs)
                self.ax3.set_title("各职业累计净利", color="w", fontsize=9, fontfamily='Microsoft YaHei')
                self.ax3.set_xticklabels(xl, fontdict={'family': 'Microsoft YaHei', 'size': 8})
                self.ax3.tick_params(colors='w', labelrotation=15, labelsize=7)
            self.canvas.draw()
        except:
            pass

    def show_detail(self, event):
        item = self.tv_agents.selection()
        if not item: return
        try:
            aid = int(self.tv_agents.item(item[0], "values")[0])
            agent = manager.agents[aid]
        except:
            return
        top = tk.Toplevel(self);
        top.title(f"{agent.name}");
        top.geometry("400x500");
        top.configure(bg="#1e1e1e")

        def lbl(r, c, t, v, col="white"):
            tk.Label(top, text=t, bg="#1e1e1e", fg="#aaa", font=("Microsoft YaHei", 9)).grid(row=r, column=c,
                                                                                             sticky="w", padx=10,
                                                                                             pady=5)
            tk.Label(top, text=v, bg="#1e1e1e", fg=col, font=("Arial", 10, "bold")).grid(row=r, column=c + 1,
                                                                                         sticky="w", padx=10, pady=5)

        tk.Label(top, text="基础信息", bg="#2d2d2d", fg="white", width=60, font=("Microsoft YaHei", 10)).grid(row=0,
                                                                                                              columnspan=4,
                                                                                                              pady=10)
        lbl(1, 0, "ID:", agent.id);
        lbl(1, 2, "流派:", ROLE_MAP.get(agent.role, agent.role))
        lbl(2, 0, "代数:", f"第 {agent.generation} 代");
        lbl(2, 2, "存活:", f"{int(time.time() - agent.last_trade)}s 前")
        tk.Label(top, text="基因序列", bg="#2d2d2d", fg="white", width=60, font=("Microsoft YaHei", 10)).grid(row=4,
                                                                                                              columnspan=4,
                                                                                                              pady=10)
        g = agent.genes
        lbl(5, 0, "杠杆:", f"x{g['lev']}", "#ffea00");
        lbl(5, 2, "仓位:", f"{g['size']:.0%}")
        lbl(6, 0, "止盈:", f"{g['tp']:.1%}", "#00e676");
        lbl(6, 2, "止损:", f"{g['sl']:.1%}", "#ff1744")
        tk.Label(top, text="财务状况", bg="#2d2d2d", fg="white", width=60, font=("Microsoft YaHei", 10)).grid(row=8,
                                                                                                              columnspan=4,
                                                                                                              pady=10)
        eq = agent.get_equity(manager.prices)
        lbl(9, 0, "权益:", f"${eq:.2f}", "#2979ff" if eq > agent.init_balance else "#ff1744")
        lbl(9, 2, "贡献:", f"${agent.total_profit_contribution:.2f}", "#00e676")
        tk.Label(top, text="持仓明细", bg="#2d2d2d", fg="white", width=60, font=("Microsoft YaHei", 10)).grid(row=11,
                                                                                                              columnspan=4,
                                                                                                              pady=10)
        tv = ttk.Treeview(top, columns=("币", "价", "盈"), show="headings", height=8)
        tv.heading("币", text="币");
        tv.heading("价", text="均价");
        tv.heading("盈", text="浮盈")
        tv.grid(row=12, column=0, columnspan=4, padx=10)
        for s, p in agent.positions.items():
            curr = manager.prices.get(s, 0)
            pnl = (curr - p['entry']) * p['amt']
            tv.insert("", "end", values=(s.upper(), f"{p['entry']:.4f}", f"${pnl:.2f}"))

# ==================================================================================
# 启动
# ==================================================================================
# ==================================================================================
# 6. 程序入口 (严格防止重复启动)
# ==================================================================================
# 全局变量占位，但不要在这里实例化！
manager = None

if __name__ == "__main__":
    print("🚀 启动银河帝国 V78 - 最终稳定版...")

    # 1. 实例化 Manager (只在这里做一次！)
    manager = Manager()

    # 2. 启动 UI
    app = Dashboard()


    def on_closing():
        print("正在关闭系统...")
        app.destroy()
        os._exit(0)  # 强制杀掉所有后台线程


    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()