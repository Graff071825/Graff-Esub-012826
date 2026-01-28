# SKILL.md · FDA 510(k) Review Studio 專家代理說明（繁體中文）

本文件說明 `agents.yaml` 中 31 個代理（Agents）的定位與建議使用方式，協助審查員在 510(k) 審查流程中快速選擇合適的 AI 工具。所有輸出預設建議使用 **繁體中文**，並可視需要補充英文原文。

---

## 1. 總覽與協調類

### 1.1 總覽協調代理（k_overall_orchestrator）
- **用途**：在多個子代理跑完後，彙整為一份「主管可讀」的高階審查摘要。
- **適用時機**：案件審查中後段，已完成風險、測試、標籤等分項分析。
- **輸入建議**：
  - 各子代理的輸出（可串接成一個長 markdown）。
  - Reviewer 的個人重點或疑慮。
- **輸出**：
  - Executive Summary、完整性評估表、決策與後續建議。

### 1.2 案件時序與里程碑整理（k_case_timeline_builder）
- 彙整各文件中的 **日期與事件**，形成案件時間軸。
- 建議在審查中後期使用，用來整理「已做什麼、還缺什麼」。

### 1.3 案件複雜度評估（k_case_complexity_scoring）
- 提供一個 **1–5 級** 的非正式「複雜度印象分數」。
- 有助於排程與選擇是否需要更多專家參與。

### 1.4 術語與縮寫整理（k_glossary_normalization）
- 為本案建立 **專有名詞與縮寫表**（中英對照）。
- 建議在案件初期執行一次，方便團隊溝通。

---

## 2. 行政與前期檢查

### 2.1 行政完整性檢查（k_admin_screening）
- 檢查是否具備基本行政文件與必要欄位（不含技術內容）。
- 可搭配 510(k) summary 與 cover letter 使用。

### 2.2 裝置與申請案總結（k_device_summary）
- 整理裝置名稱、分類、適用族群與主要議題。
- 適合作為 **審查起點** 的「案件總攬」。

---

## 3. 臨床定位與實質等同性

### 3.1 用途與適應症審查（k_intended_use_indications）
- 比對 **Intended Use** 與 **Indications for Use** 的一致性。
- 找出「隱含新適應症」或族群/環境明顯不同之處。

### 3.2 裝置描述與技術特性（k_device_description_techchars）
- 整理裝置結構、作動原理與關鍵技術特性。
- 給後續 SE 比較與風險分析使用。

### 3.3 前例裝置識別（k_predicate_identification）
- 彙整申請人提出的 predicate 裝置（510(k) 號碼、名稱、適應症概要）。
- 明確區分「主要比較 predicate」與次要參考裝置。

### 3.4 實質等同性比較（k_se_comparison）
- 建立 **適應症** 與 **技術特性** 的對照表。
- 標示差異及其可能對安全/有效性的影響。

---

## 4. 風險與安全性

### 4.1 風險管理與 ISO 14971（k_risk_management_iso14971）
- 建立 hazard / 失效模式 / 控制措施 / 殘餘風險表。
- 指出風險覆蓋的「明顯缺口」。

### 4.2 紅旗與高風險警示（k_red_flag_detector）
- 針對大量文件進行快速「紅旗掃描」。
- 適合在案件初期對長文檔做粗篩，找出須特別留意的議題。

### 4.3 兒科與易受害族群（k_pediatric_use）
- 指出是否涉及 **兒科、孕婦、高齡** 等族群，及其額外風險。

---

## 5. 測試與驗證相關代理

### 5.1 性能與台架測試（k_bench_testing）
- 以表格形式整理所有 bench test 的方法、樣本量與結果。
- 有助於快速檢查「測試是否覆蓋主要風險面」。

### 5.2 生物相容性（k_biocompatibility）
- 依據接觸類型與時間，比對應有的 ISO 10993 試驗。
- 清楚列出已提供與欠缺的試驗/論證。

### 5.3 滅菌與無菌保證（k_sterilization）
- 整理滅菌方法、SAL、驗證方式與關聯包裝試驗。

### 5.4 包裝完整性與效期（k_packaging_shelf_life）
- 檢視包裝驗證與 shelf life 宣稱是否有足夠試驗支持。

### 5.5 EMC 與電氣安全（k_emc_electrical）
- 整理 IEC 60601/61010/EMC 測試與其結果。
- 檢視是否與預期使用環境相符（例如家庭 vs 醫院）。

### 5.6 軟體驗證與 SOUP（k_software_validation）
- 檢查是否有充分的軟體驗證與 SOUP 管理資訊。
- 特別適用於含嵌入式軟體或 SaaS / cloud component 的裝置。

### 5.7 資安與連網安全（k_cybersecurity）
- 專門針對 **連網醫材** 的威脅模型與控管措施。
- 參考 FDA Cybersecurity guidance。

### 5.8 人因與可用性（k_usability_hfe）
- 建立「使用情境–使用錯誤–後果–控制措施」對照。
- 適用於複雜 UI、家庭使用或高風險操作步驟。

---

## 6. 臨床與 IVD 專業

### 6.1 臨床資料與文獻（k_clinical_data）
- 梳理臨床研究設計、結果與對 SE 的貢獻。
- 包含非正式的真實世界證據（RWE）整理。

### 6.2 統計與樣本數（k_statistics）
- 評估試驗的統計假設與樣本數合理性。
- 不進行正式再分析，但指出方法學弱點。

### 6.3 Post-market 經驗（k_postmarket_surveillance）
- 若有既上市裝置資料，整理不良事件或 RWE 訊息。

### 6.4 IVD 專家（k_ivd_specific）
- 專為體外診斷醫材（IVD）設計，聚焦 analytical/clinical performance。
- 也可檢視樣本型態與干擾因子評估。

### 6.5 兒科/易受害族群（k_pediatric_use）
- 同上節說明，特別針對族群風險。

### 6.6 複合產品（k_combination_products）
- 整理含藥/生物成分的複合產品議題。
- 協助判斷是否可能需要與其他中心協調（高階描述）。

---

## 7. 標籤、IFU 與設計管制

### 7.1 標籤與 IFU / UDI（k_labeling_ifu_udi）
- 梳理警語、禁忌症、注意事項與操作步驟。
- 檢查是否與適應症與風險控制 **一致**。

### 7.2 設計管制與 QSR 連結（k_qsr_design_controls）
- 高階查看輸入中是否提到設計驗證/確認等資訊。
- 不做 QMS 稽核，而是標示資訊是否充分。

---

## 8. 指引對照與溝通文件

### 8.1 FDA 指引對照與落差（k_guidance_alignment）
- 將案件資料與 **指定的 FDA guidance/標準** 建立 checklist 對照。
- 輸出符合/部分/不清楚/不符合的狀態表。

### 8.2 缺失/補件信草擬（k_deficiency_rfi_letter）
- 根據審查結果生成缺失/補件信草案（RFI / AI letter）。
- 語氣專業、具體，利於後續人工修訂。

### 8.3 申請人回覆評估（k_sponsor_response_review）
- 逐項檢查申請人回覆是否充分解決原疑慮。
- 建議是否可關閉該議題，或仍需進一步問題。

### 8.4 決策備忘錄草稿（k_decision_memo）
- 協助撰寫 Decision Memo 初稿，整合支持與反對 SE 的主要論點。

---

## 9. 內部知識與訓練

### 9.1 內部訓練與知識萃取（k_internal_training_note）
- 將本案經驗去識別化，整理為內部訓練教材。
- 含簡短小測驗題目，方便培訓新進 reviewer。

### 9.2 中英雙語協助（k_translation_bilingual_helper）
- 協助中英雙向翻譯技術與審查文字，保持術語一致。
- 建議搭配 `k_glossary_normalization` 使用。

### 9.3 審查提示詞精煉（k_prompt_refiner）
- 將 reviewer 的需求轉寫成更清楚的 Prompt（中英版本）。
- 適合想要「客製化進階提示」時使用。

---

## 10. 建議使用流程（範例）

1. **案件起始**
   - `k_device_summary` → `k_glossary_normalization` → `k_case_complexity_scoring`
2. **臨床定位與 SE 初步**
   - `k_intended_use_indications` → `k_device_description_techchars`
   - `k_predicate_identification` → `k_se_comparison`
3. **安全與測試覆蓋**
   - `k_risk_management_iso14971` + `k_bench_testing` + `k_biocompatibility`
   - 視需要加入 `k_emc_electrical`、`k_software_validation`、`k_cybersecurity`
4. **標籤與人因**
   - `k_labeling_ifu_udi` + `k_usability_hfe`
5. **指引與缺口**
   - `k_guidance_alignment` → `k_red_flag_detector`
6. **溝通與決策文件**
   - `k_deficiency_rfi_letter` → (收到回覆後) `k_sponsor_response_review`
   - 最後使用 `k_decision_memo` 與 `k_overall_orchestrator` 彙整。

---

## 11. 一般注意事項

- 所有代理輸出 **不構成法律意見**，僅為技術/科學層面之輔助分析。
- 審查員應對 AI 輸出進行人工判讀與修正，特別是：
  - 具爭議性的風險判斷
  - 新穎技術或族群相關議題
  - 法規與管轄權相關結論
- 建議在內部系統中標明：AI 輸出為草案或參考，不可直接視為正式審查結論。

---

如需新增或調整代理，建議：
- 避免過度重疊的職責。
- 明確界定每個代理的「輸入預期」與「輸出格式」。
- 盡量使用表格與分節標題，方便後續在 Streamlit 介面中閱讀與再利用。
