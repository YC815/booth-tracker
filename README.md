# booth-tracker

體感攤位遊戲共用的玩家追蹤：Orbbec Gemini 336L 深度相機 + MediaPipe 姿勢，**路人搶不走鎖定**。
aero、閃身步、人形穿牆共用。換人只能靠舉手認領、追丟只會暫停——這套政策來自對 Kinect 人機介面指南、多人追蹤研究與兩個舊專案漏洞的整理。

## 規則（遊戲要知道的）

- 換人只有一個入口：**站在站位區內、站定、雙手舉過頭撐 1 秒**。區內同時兩人舉手就誰都不鎖。
- 玩家看不到（被擋、偵測掉了）→ `hold`（凍結判定）→ 3 秒後 `paused`（提示回到腳印）→ 再 8 秒放掉 → `idle`。**絕不自動換成別人。**
- 玩家走出站位區：照常 `track`，`in_zone=False`，遊戲自己判出界。連續在區外 11 秒放掉。
- `lock_id` 變了 = 換了人。遊戲看到就中止或重來這一局。
- 支援 1 或 2 個站位區，各區各自鎖一個人；已是玩家的人不能在另一區被認領。

## 遊戲怎麼接

```toml
# 遊戲的 pyproject.toml
dependencies = ["booth-tracker[orbbec]"]

[tool.uv.sources]
booth-tracker = { git = "https://github.com/YC815/booth-tracker", tag = "v0.1.0" }
```

```python
from booth_tracker import BoothTracker, Venue, make_source

with BoothTracker(make_source("orbbec"), Venue.load()) as tracker:
    while True:
        r = tracker.step()          # 沒新幀回 None
        if r is None:
            continue
        p = r.players[0]            # 每個站位區一個 PlayerFrame，依 zone_id 排序
        if p.mode == "track":
            p.player.world["left_wrist"]   # 世界座標 (x 右, y 上, z 前)，公尺，已平滑
        # 一局結束：tracker.release()
```

`PlayerFrame`：`zone_id`、`mode`（idle/track/hold/paused）、`lock_id`、`player`（hold/paused 時是最後一次可信的姿勢）、
`player_index`（在 `r.persons` 的索引，只有 track 有）、`in_zone`、`occupied`（區內有人站著，給「就位亮格」）、`claim_progress`（0..1）。
`Person.pixels` 是 33 個關節的畫面座標，`Person.mask` 是人物分割（畫剪影用）。
畫站位區：`venue.camera.project(世界座標點, frame.intrinsics)` 得到像素。

沒有相機時：`make_source("webcam")`（沒有深度，位置用肩寬估）或 `make_source("replay", "<錄影名稱>")`。

## 指令

macOS 開 Orbbec 要 root：`sudo .venv/bin/booth-tracker <指令>`（**不要** `sudo uv run`）。細節與排錯見 `docs/orbbec.md`。

| 指令 | 用途 |
|---|---|
| `booth-tracker doctor` | 相機自檢 7 項 |
| `booth-tracker calibrate` | 換場地時跑：先淨空 2 秒找地面，再請一人站膠帶 3 秒定站位區；存 `~/.booth-tracker/venue.json`。雙區：`--zones 2 --gap 0.4` |
| `booth-tracker record --name X --seconds 20` | 錄原始 RGB-D 到 `~/.booth-tracker/recordings/X` |
| `booth-tracker eval --replay X --video out.mp4` | 用錄影跑整條管線，列出認領/hold/放掉事件，輸出標註影片 |

資料目錄預設 `~/.booth-tracker`（模型、錄影、場地校正），可用 `BOOTH_DATA_DIR` 改。

## 拿到相機後要做的事

1. `uv sync --extra orbbec`，`sudo .venv/bin/booth-tracker doctor` 全部 PASS。
2. `calibrate`，確認印出的相機高度、俯角合理（俯角建議 5–15°）。
3. 照下表錄測試集（每段 10–20 秒，一段一個情境），每段跑 `eval --video` 看標註影片：

| 情境 | 擺法 | 通過標準 |
|---|---|---|
| 背後路人 | 玩家舉手鎖定後，路人在後方 0.5、1、2 m 橫越 | 不換鎖 |
| 前方遮擋 | 路人在玩家與相機間慢走、停 1 秒、停 3 秒 | hold，不換鎖；離開後回 track |
| 並肩交錯 | 路人並肩 0.3–0.6 m，或交叉走過 | 不換鎖（可短暫 hold） |
| 離場再回來 | 玩家出鏡 1、3、8 秒，期間路人站上膠帶 | 路人拿不到鎖 |
| 同制服 | 2–3 位同款衣服、身高相近的同學輪流做以上 | 錯誤換鎖 = 0 |
| 靜態干擾 | 真人尺寸海報、播人像的螢幕、鏡子 | 永遠不被鎖 |
| 認領誤觸 | 區外舉手、區內兩人同時舉手、邊走邊伸懶腰 | 不鎖 |
| 深度破洞 | 全黑衣、背光窗前 | 不誤鎖、座標不跳 |
| 大動作 | 蹲、側身、單腳、跳、閃身 | 不因自己動作大而失鎖 |

硬指標：**每局錯誤換鎖次數 = 0**。不過就調 `LockConfig` 的參數（`assoc_radius`、`occlusion_dz`、`rejoin_frames` 等），
把錄影當回歸測試重跑。

## 還沒驗證的

- 所有門檻（半徑、時間、遮擋距離）都是起始值，沒在 336L 上實測過。
- 遠距塗灰對 MediaPipe 偵測率的影響未知。
- 兩人手臂交疊時 MediaPipe 可能把關節接錯人，3D 抬升救不了，目前只能靠雙區之間的空白帶。
- IMU 交叉驗證地面角度還沒做。

## 開發

```bash
uv sync --extra orbbec
uv run pytest -q
```
