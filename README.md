# cover-ai

音色转换与翻唱的脚本集。所有工具都通过 `run.py` 一个入口调用。

## 快速上手

```powershell
git clone git@github.com:zxv14/cover-ai.git
cd cover-ai

# 1. 装依赖（torch 必须单独从 cu128 索引装）
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
pip install -r requirements.txt --no-cache-dir

# 2. 看有哪些工具
python run.py list

# 3. 典型流程：干声 → 切片 → 筛选 → 训练 → 推理 → 混音
python run.py slice  --use-separated work/vocal --out work/dataset
python run.py filter --exp rvc/logs/myvoice --threshold -45
python run.py infer  --repo tools/DDSP-SVC-6.3 --model exp/myvoice/model_30000.pt `
                     --input dry.wav --output converted.wav --step 100
python run.py mix    --vocal converted.wav --backing inst.wav --out final.wav
```

每个子命令的参数和对应脚本完全一致，`python run.py mix --help` 可以看说明。

**前置**：Python 3.10 + `torch 2.7.1+cu128`。cu128 是硬要求——50 系显卡用 cu121/cu124 会报
`no kernel image is available`。

---

## 一、先确认环境

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
```

要看到 `2.7.1+cu128` 和 `(12, 0)`。不对就别往下走。

---

## 二、数据准备

### 1. 切片 + 质检

把干声放进一个目录（每个文件一条音频），切成训练片段：

```powershell
python run.py slice --use-separated <干声目录> --out <数据集目录> `
  --min-sec 2.5 --max-sec 10 --val-clips 30
```

产出 `<数据集目录>\train\audio` 和 `val\audio`。

### 2. 纯净度筛选（RVC 用）

带伴奏残留的片段会污染音色模型。按"静音段电平"筛掉：

```powershell
python run.py filter --exp <rvc实验目录> --threshold -45
```

它读 `0_gt_wavs` 里每段的第 10 百分位电平，低于阈值的留下，其余移到 `_rejected` 子目录（不删），然后生成 RVC 需要的 `filelist.txt`。

**注意**：这个指标对"连续演唱、中间没有停顿"的片段无效（测不到间隙）。这类片段应该按分离器来源放行——MelBand 分离出来的可以直接信。

### 3. 生成 DDSP 训练配置

```powershell
python run.py config --name <实验名> --train <数据集>\train --val <数据集>\val
```

输出 `<DDSP仓库>\configs\train-<实验名>.yaml`，并在 `exp/<实验名>` 建实验目录。

**Windows 上必须手动改两个值**，否则训练会卡死或爆内存：

```yaml
data:
  num_workers: 0
  cache_device: cuda
train:
  num_workers: 0
  cache_device: cuda
```

---

## 三、训练

### DDSP-SVC

```powershell
python run.py patch --repo <DDSP仓库>      # 只需跑一次
cd <DDSP仓库>
python preprocess.py -c configs/train-<实验名>.yaml -j 2
python train_reflow.py -c configs/train-<实验名>.yaml
```

`patch_train_windows.py` 修的是 `DotDict` 缺 dunder 方法导致 Windows 下多进程 pickle 崩溃的问题——不修的话 `preprocess.py` 会报 `EOFError: Ran out of input`。

### 监控

```powershell
python run.py status --exp <DDSP仓库>\exp\<实验名>
python run.py watch --exp <DDSP仓库>\exp\<实验名>
```

`watch_train.py` 会检查验证 loss 趋势、NaN、检查点轮换，异常时退出码为 1，可以当门禁用。

---

## 四、推理 + 混音

```powershell
python run.py infer --repo <DDSP仓库> --model <检查点.pt> `
  --input <干声.wav> --output <转换.wav> --key 0 --step 100

python run.py mix --vocal <转换.wav> --backing <伴奏.wav> --out <成品.wav> `
  --vocal-db 2 --limit --limit-range-db 8
```

- `--step` 默认 50，**建议 100 以上**。步数低会留噪声，听感发"电"。
- `--vocal-db` 是相对伴奏的响度偏移；`--limit` 开限幅器，`--limit-range-db` 控制作用范围。
- 输出做了峰值归一化到 -1 dBFS，**没有做母带**，成品链（EQ/压缩/限幅）留给你自己。

---

## 五、评估

```powershell
python run.py follow --original <干声.wav> --converted <转换.wav>
python run.py speaker --target-dir <她的片段目录> --audio <转换.wav>
python run.py eval --exp <DDSP仓库>\exp\<实验名> --tag <标签>
```

### 怎么读这些数字

`check_vocal.py` 给的是**包络相关**：转换结果与原唱在时间轴上是否同步。`> 0.9` 说明跟得紧，和音色无关。

`speaker_id.py` 给的是**说话人嵌入余弦相似度**，会先打印目标说话人的内部一致性基线（同一个人不同片段之间，通常 0.98+）。判断标准是看转换结果离这个基线多远，**不要看绝对值**——不同说话人之间本身就有 0.74 左右的相似度。

`eval_model.py` 是前两个的封装，一条命令跑完并给出判定。

**两个别踩的坑**：

1. **别只看训练 loss**。验证 loss 平了但音色可能还在改善，反之亦然。
2. **别用频谱距离代替"像不像"**。频谱包络距离被录音条件主导（同一个话筒、同一个房间会让它虚高），它不反映"这个人是谁"。早期我们用这个指标得到过 +7.9dB 的"改善"，但听感完全不像。

---

## 六、检查模型能不能用

拿到别人的 `.pt` / `.pth` 先用这个过一遍：

```powershell
python run.py check --model <模型文件> --repo <DDSP仓库>
```

它会走真实的推理加载路径，检查 encoder、vocoder、config 是否配套。**DDSP 6.2 和 6.3 的检查点不通用**，跨版本混用会报 `RuntimeError: Error(s) in loading state_dict`。

---

## 七、上游依赖

本仓库只是一层胶水脚本。**真正的模型训练和推理都由这些项目完成**，它们各自独立维护：

| 项目 | 在这里的作用 | 地址 |
|---|---|---|
| **DDSP-SVC** | 音色转换模型（用 `6.3` 分支） | https://github.com/yxlllc/DDSP-SVC |
| **RVC** | 检索式音色转换（音色通道更强） | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI |
| **ContentVec** | 内容特征提取（去说话人信息） | https://huggingface.co/lengyue233/content-vec-best |
| **RMVPE** | 音高（f0）提取 | https://github.com/yxlllc/RMVPE |
| **NSF-HiFiGAN** | 声码器 | https://huggingface.co/None1145/DDSP-SVC-Base |
| **MelBand-Roformer** | 人声分离| https://huggingface.co/Kim-ML/mel_band_roformer |

`setup.ps1` 会自动克隆前两个，剩下的底模它会按需下载。

**遇到问题请先看上游的 issue 和文档**——这个仓库的脚本只负责串流程，模型本身的行为由上游决定。上面那张报错表只覆盖 Windows 安装层面的坑。

各项目的许可证见其仓库内的 LICENSE 文件，使用前请自行确认。

