# 整合式 AI 研究系統｜專案總覽

## 1. 系統目標
- GCP 為控制器（Cloud Run + Flask API），Kaggle 為計算節點，GCS 為長期儲存與主要交換通道，ngrok 為臨時互動備援。
- 分析架構（Onion）：Phenomena → Impact → Constraints → Direct Cause → Root Mechanism → Verification。
- 七個研究迴圈：Problem Definition、Evidence/Baseline、Hypothesis、Experiment Design、Execution、Analysis、Replication/Audit/Delivery。
- Dispatcher 演算法選擇：UCT(i) = W_i/N_i + c√(ln N_parent/N_i)。
- Shadow Brain：反向盲點提問、證據與邏輯審核、anchor point 驗證。
- 技能階層：Gene → Skill → Workflow；五大類技能；技能代碼格式 CATEGORY_SKILLNAME。
- 技能庫上限 200 個 Active skills；四維評估（Stability、Composability、ROI、Uniqueness）；Thompson Sampling；Champion vs. Challenger；生命週期 Candidate → Active → Dormant → Retired。
- 資料管理：原子切片、fingerprints、向量／圖索引、TurboQuant；ZIP 無損封裝、SHA-256、Merkle Root；高價值 artifacts 加密。
- 失敗需記錄、微調、連結 retry tickets；目標為從最後已驗證 checkpoint 自動恢復。
- 保留期限：操作 logs 60 天；研究紀錄永久保存。

## 2. 工作規則
- 以 V1 為基線，逐檔更新，每次只修改一個檔案，版本逐檔遞增。
- 未變更的檔案沿用既有版本與驗證結果，不重複審閱。
- 每次交付完整新版，並附上檔名、版本、修改摘要、相依影響、審閱與測試狀態。
- 未實際執行的測試一律標記為「未測試／待驗證」，不得宣稱通過。
- 基線檔案路徑未提供的，不得臆測。

## 3. 檔案清單與版本（待確認項標註 ⚠）
- shared/contracts.py — V2 ⚠（截圖顯示 V2，交接記憶為 V1，待核對內容與缺少的型別）
- shared/skill_genome.py — V1（Group 1 部分審閱）
- shared/integrity.py — V1（Group 1 部分審閱；見第 4 節待補項）
- shared/research_record.py — V1（Group 1 部分審閱）
- shared/checkpoint_store.py — V1（Group 1 部分審閱）
- config/system.yaml — V2 ⚠（截圖 V2，交接記憶為 V1；環境變數展開與驗證方式待確認）
- config/market-sources.yaml — V2 ⚠（截圖 V2，交接記憶為 V1；與 market_research V2 的整合待確認）
- config/skill-catalog.json — V1（Group 2 已審閱）
- gcp/research_flow.py — V1（Group 2 已審閱）
- gcp/shadow_audit.py — V2（已審閱，測試未確認）
- gcp/market_research.py — V2（已審閱，測試未確認）
- gcp/app.py — V2 ⚠（截圖 V2，交接記憶為 V1）
- gcp/wsgi.py — V2 ⚠（截圖 V2，交接記憶為 V1）
- gcp/kaggle_dispatch.py — V1（Group 3 已審閱）
- gcp/requirements.txt — V1（Group 3 已審閱）
- gcp/deploy/Dockerfile — V1（Group 5 已審閱）
- gcp/deploy/cloudbuild.yaml — V1（Group 5 已審閱）
- kaggle/worker.py — V2（尚未在 Kaggle／GCS 實測）
- kaggle/worker.ipynb — V1（Group 4 已審閱）
- kaggle/generate_kernel_metadata.py — V1（Group 4 已審閱）
- kaggle/requirements.txt — V2（尚未安裝測試）
- INSTALL.md — V2（尚未在實際環境驗證）
- 基線檔案 ×3 — V1，路徑未提供 ⚠
- gcp/llm_client.py — 是否存在未確認 ⚠

## 4. 待驗證項目
1. shared/contracts.py 缺少的型別與相依介面核對。
2. integrity.py 的已知限制：尾部截斷無法偵測；並發追加可能分叉；哈希依賴 Python 型別（1 與 1.0）；created_at 未檢查格式與順序。
3. Worker 本地 events.json 鏈是否整合進永久研究紀錄。
4. Kaggle notebook 是否讀取 Kaggle Secret 並設定 GCP_SERVICE_ACCOUNT_JSON 環境變數。
5. Kaggle 到 GCS 的端到端認證、dispatch 讀取、結果與 artifacts 寫回。
6. Worker 結果如何同步更新 GCP controller 的 job status。
7. market_research.py V2 與 market-sources.yaml 的實際整合。
8. system.yaml 的環境變數展開與驗證方式。
9. 具體 research handlers 尚未提供或驗證。
10. 自動尋找最後已驗證 checkpoint 的完整恢復流程未端到端驗證（目前僅依 GCS_CHECKPOINT_OBJECT 下載指定物件）。
11. Cloud Build substitutions、secret mapping、Cloud Run service account、region、service 名稱需對照實際 YAML。
12. Cloud Run IAM 驗證與應用程式 Bearer token 需共同測試；X-Serverless-Authorization 傳遞方式未實測。

## 5. 每次對話的流程
1. 附上本文件與要處理的單一檔案（含其直接相依的介面）。
2. 說明本次目標，例如「只處理 shared/contracts.py」。
3. 交付完整新版，附版本、修改摘要、相依影響與測試狀態。
4. 由使用者提交到 repo，並更新本文件的版本表與 CHANGELOG。

## 6. 下一步
- 核對 shared/contracts.py 的實際內容，確認 V2 是否已完成。
- 完成後依相依順序處理 shared/core 模組。

---

# CHANGELOG

## 格式
- 日期｜檔案｜版本｜修改摘要｜相依影響｜測試狀態

## 紀錄
- 2026-11-12｜PROJECT.md｜— ｜建立專案總覽與版本表初版｜—｜不適用
