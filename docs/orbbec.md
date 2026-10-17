# Orbbec Gemini 336L 接入說明

程式：`backend/src/aero/sources/orbbec.py`（FrameSource）、`backend/src/aero/doctor.py`（診斷）。

## 安裝（已驗證：PyPI wheel 可用）

- PyPI 套件名是 **`pyorbbecsdk2`**（import 名稱仍是 `pyorbbecsdk`）。2.1.2（2026-08-11）有 `cp312-cp312-macosx_13_0_arm64` wheel；2.0.18 沒有 macOS wheel。
  來源：<https://pypi.org/project/pyorbbecsdk2/>（JSON API 檔案清單）、<https://github.com/orbbec/pyorbbecsdk>（`v2-main` 分支 README：支援 macOS，測試環境 M2 / macOS 13.2；SDK v2 才支援 Gemini 330 系列）。
- 本專案：`uv add --optional orbbec pyorbbecsdk2` 已加入 `pyproject.toml`（extra 名 `orbbec`）。
  - 安裝：`cd backend && uv sync --extra orbbec`；執行：`uv run --extra orbbec ...`。
  - 本機 macOS arm64 / Python 3.12 實測 `import pyorbbecsdk` 成功，`get_version()` = 2.9.3。
- 若將來 wheel 不能用：從 GitHub Releases 取 wheel，或照 <https://github.com/orbbec/pyorbbecsdk> README 從 source build（本專案沒走過這條路，步驟未驗證）。`orbbec.py` 匯入失敗時的錯誤訊息會指向本文件。

## macOS 必須 root（最大風險）

macOS 的 `UVCAssistant` 常駐程式會獨佔所有 UVC 裝置，Orbbec 相機（Gemini 330 為 UVC 裝置）沒有 root 時 SDK 開不了：
症狀是 `uvc_open ... already opened` 或串流啟動後卡住。官方唯一說法是 `sudo` 為暫時解法。

- 來源：<https://github.com/orbbec/OrbbecSDK_v2/issues/124>（Femto Mega 的案例，Orbbec 成員回覆 sudo 是暫時解法，尚未修）；社群設定紀錄 <https://gist.github.com/rootnotez/72b0d0ac0f82093689867b5f8338671b>（`uvc_open already opened` → 用 sudo）；OpenCV 的 Orbbec UVC 教學也寫 macOS 需 sudo（<https://docs.opencv.org/4.13.0/d1/d33/tutorial_orbbec_uvc.html>，只看到搜尋摘要，頁面本身 403）。
- 上述來源的案例是 Femto Mega 與 OrbbecViewer；**沒有找到 Gemini 336L 在 macOS 上的明確紀錄**，只能推論同為 UVC 所以同樣受影響。
- 做法：`sudo backend/.venv/bin/python -m aero.doctor`。**不要用 `sudo uv run`**（會讓 uv cache 變 root 擁有）。doctor 以 root 存檔時會 chown 回 `SUDO_UID`。
- 同一時間只能有一個程式開相機：先關 OrbbecViewer。
- 官方 SDK 的 `setup_env.py` 寫 macOS 不需額外環境設定；Linux 才需要 udev rules（`99-obsensor-libusb.rules`）。

## 相機

- Gemini 336L：USB 3.0 Type-C，深度最高 1280x800@30（<https://orbbec.com/products/stereo-vision-camera/gemini-336l>）。必須走 USB 3.x 線與埠；USB 2.0 辨識要約 20 秒且頻寬不足（<https://doc.orbbec.com/documentation/Orbbec%20Gemini%20330%20Series%20Documentation/Troubleshooting>）。
- 韌體：OrbbecSDK_v2 README 的表列 Gemini 330 / 336L 最低 1.2.20、建議 1.8.10（表格抓取文字不易讀，需對照原頁確認）。doctor 只對低於 1.2.20 提出警告。
- SDK 隨 wheel 內附 `config/OrbbecSDKConfig.xml`，其 `<Gemini336L>` 預設為 depth 848x480@30 `Y16`、color **1280x720@30 `MJPG`**。因此 1280x720 color 在 336L 上確定存在，預設格式是 MJPG。

## API 對照（皆取自 wheel 內附的 `pyorbbecsdk.pyi` 與 `examples/`，這些檔案與 repo `v2-main` 的 `examples/` 相同）

| 需求 | 用法 | 依據 |
|---|---|---|
| 開 stream | `Pipeline()`、`pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)`、`profile_list.get_video_stream_profile(w, h, fmt, fps)`（0 = 不限）、`profile_list.get_default_video_stream_profile()`、`config.enable_stream(profile)`、`pipeline.start(config)` | `examples/beginner/03_color_and_depth_aligned.py`、`04_camera_calibration.py` |
| 每幀 color+depth 齊全 | `config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)` | 同上 03 的 SW 路徑 |
| 取幀 | `pipeline.wait_for_frames(1000)`，逾時回 `None`；`frames.get_color_frame()` / `get_depth_frame()` | `quick_start.py` |
| D2C 對齊 | 軟體：`AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM).process(frames)`，回傳仍是 frameset。硬體另有 `config.set_align_mode(OBAlignMode.HW_MODE)` 搭配 `pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)`，裝置不支援時要自己退回軟體 | 03 範例；本專案選軟體 |
| 深度單位 | `depth_frame.get_depth_scale()` 乘上 uint16 原始值得到 **毫米**（範例的 `MIN_DEPTH = 20  # mm`）；本專案再 /1000 成公尺 | `quick_start.py`、03 |
| color 轉 RGB | `frame.get_format()` 判斷：`RGB`（直接 reshape）、`MJPG`（`cv2.imdecode`，得 BGR）、`YUYV`/`UYVY`/`NV12`/`NV21`/`I420`（OpenCV `COLOR_YUV2RGB_*`） | `examples/utils.py` 的 `frame_to_bgr_image` |
| color 內參 | `pipeline.get_camera_param().rgb_intrinsic`（`fx fy cx cy width height`），需在 pipeline 啟動並拿到第一幀後呼叫；備援 `color_frame.get_stream_profile().as_video_stream_profile().get_intrinsic()` | 04 範例、`pyorbbecsdk.pyi` |
| 裝置資訊 | `Context().query_devices()` → `get_count()`、`get_device_by_index(i).get_device_info()` → `get_name() / get_serial_number() / get_firmware_version() / get_pid() / get_connection_type()` | `pyorbbecsdk.pyi`、03 範例開頭 |

## 本專案的行為

- color：請求 1280x720@30，格式依序嘗試，找不到再放寬 fps；全找不到時錯誤訊息列出裝置實際提供的 profile（寬x高 格式 @fps）。預設順序 MJPG、RGB、YUYV、任意。
  - **環境變數 `AERO_ORBBEC_COLOR=mjpg|rgb`（預設 `mjpg`）**：設成 `rgb` 則優先請求 RGB（官方軟體 D2C 範例用的是 RGB）。若 MJPG 搭配 AlignFilter 出現 depth 全 0 或對齊錯位，明天先改用 `AERO_ORBBEC_COLOR=rgb` 重跑 doctor（root 下記得帶變數：`sudo AERO_ORBBEC_COLOR=rgb backend/.venv/bin/python -m aero.doctor`）。
- depth：先用 SDK 預設 profile（336L 預設 848x480@30 Y16，解析度小，軟體 D2C 才跑得動）；預設不是 Y16 才搜尋 Y16 profile。不主動搜尋，因為搜尋可能挑到 1280x800 而拖慢 fps。
- 對齊後 depth 若尺寸與 color 不同，以最近鄰縮放到 color 尺寸。
- 內參：解析度與 color 幀一致就直接用；長寬比相同就按比例縮放 fx/fy/cx/cy 並更新 width/height（log 警告）；長寬比不同則 `Frame.intrinsics = None` 並只記錄一次 error（不回傳錯的內參）。doctor 第 5 項會 FAIL。
- 斷線判定：`wait_for_frames` 丟例外，或連續超過 3 秒沒有完整幀。之後 `read()` 回 `None`，每 2 秒關掉舊 pipeline 並重開新的。
- `start()`：`pipeline.start` 包在 daemon thread，20 秒逾時。逾時不丟例外，`last_error` 與 `start_timed_out` 記錄原因，`read()` 之後每 2 秒重試；舊的 start 還沒返回時不會再開第二個 pipeline，返回後才關掉舊 pipeline 再重開。開相機失敗同樣只記錄在 `last_error`；只有 SDK 匯入失敗才丟 `OrbbecSdkMissing`。

## 診斷

```
sudo backend/.venv/bin/python -m aero.doctor      # 有相機（macOS）
cd backend && uv run --extra orbbec python -c "from aero.doctor import run; raise SystemExit(run())"   # 無相機也能跑，預期第 2 項 FAIL
```

逐項印 `[PASS]` / `[FAIL] 原因`，第一個 FAIL 之後標 `[SKIP]`。輸出 `data/doctor.png`（color 與 depth colormap 50% 疊圖）。

| 症狀 | 可能原因 |
|---|---|
| 第 2 項 FAIL：沒有裝置 | 線材/埠非 USB 3.x、別的程式佔用、剛插上需等 30 秒、macOS 未 root |
| 第 3 項 FAIL 或逾時 | macOS 未 root（`uvc_open already opened`）、profile 不存在（訊息會列出可用 profile）、相機被占用 |
| 第 4 項 fps 低 | USB 2.0、CPU 吃緊（軟體 D2C 在 CPU 上）；必要時評估硬體 D2C |
| 第 5 項內參解析度不符 | `get_camera_param` 對應的 profile 與實際 color 不同，3D 座標會偏 |
| 第 6 項有效深度低 | 目標太近（< 0.3 m）、黑色/反光表面、對齊失敗 |
