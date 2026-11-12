"""
Infrastructure - Skill System with 250 Elite Skills.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from enum import Enum
import json
import asyncio

from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import SkillError, ValidationError
from ..shared.observability import get_logger, get_metrics, traced, metered
from ..domain.entities import (
    SkillId, SkillCategory, GeneKey, SkillSpec, AgentCapability,
    AgentId, AgentRole
)
from ..application.ports import OpenSourceSearchPort, LLMProviderPort, SandboxPort


logger = get_logger("infrastructure.skills")
metrics = get_metrics()


# 250 Elite Skill Gene Keys
ELITE_SKILL_GENES = {
    SkillCategory.INFORMATION_RETRIEVAL: [
        "即時全網深度檢索", "多搜尋引擎結果聚合與去重", "特定網域與垂直資料庫精準搜尋",
        "學術論文與arXiv最新預印本檢索", "全球專利文獻檢索與技術點提煉",
        "政府公開數據與法規條文檢索", "社交媒體即時熱點與趨勢追蹤", "競爭對手官網動態與價格變更監控",
        "供應商與企業工商背景調查檢索", "跨語言海外市場資訊同步檢索",
        "企業內部知識庫問答(RAG)", "跨文件(PDF/Word/PPT)語義檢索", "歷史專案檔案與經驗教訓檢索",
        "客戶服務歷史工單與解決方案匹配", "產品規格書與技術手冊精準定位",
        "法律合約範本庫快速檢索與比對", "財務制度與報銷規範自動查詢", "程式碼倉庫(Codebase)全局語義檢索",
        "多模態文件檢索(同時搜尋文件內的文字與圖表)", "帶時間權重的知識檢索(優先採信最新版本文件)",
        "長文本結構化核心摘要", "萬字級書籍或報告的章節大綱提煉", "非結構化文本轉標準JSON/CSV格式",
        "發票、收據與單據關鍵欄位自動提取(OCR+LLM)", "履歷表(CV)核心資歷與技能標籤提取",
        "商業合約關鍵條款(金額、期限、違約責任)提取", "網頁正文深度清洗與廣告雜訊過濾",
        "複雜PDF跨頁表格精準還原與提取", "命名實體識別(自動標記人名、地名、機構、產品名)",
        "文本關鍵字與主題標籤(Auto-Tagging)自動生成",
        "多來源事實核查(Fact-Checking)與交叉驗證", "資訊衝突與數據矛盾點自動標記",
        "情報可信度與來源權威性評級", "多份同主題報告的觀點合併與去重",
        "知識圖譜(Knowledge Graph)實體關係自動建構", "會議與訪談逐字稿發言者區分與重點提煉",
        "碎片化筆記自動歸納為結構化知識卡片", "領域專屬術語庫(Glossary)自動提取與維護",
        "跨語言專業文獻精準本地化翻譯", "雙語對照文件自動排版與術語一致性檢查",
        "過時與失效知識自動識別與歸檔", "知識庫盲區檢測(分析AI無法回答的高頻問題)",
        "長期對話記憶壓縮與關鍵用戶偏好提取", "跨會話(Cross-Session)用戶畫像動態更新",
        "文件自動分類與資料夾層級歸檔", "敏感資訊與個人隱私(PII)自動識別與脫敏",
        "檔案版本差異比對(Diff)與變更摘要", "每日行業情報簡報自動彙編",
        "問答對(QA Pairs)數據集自動生成(用於微調模型)", "引用來源精準溯源與超連結自動標註"
    ],
    SkillCategory.LOGICAL_REASONING: [
        "自然語言轉SQL查詢(Text-to-SQL)", "複雜多表關聯(JOIN)資料庫查詢優化",
        "自動化Python/Pandas數據清洗腳本執行", "數據缺失值、重複值與異常值(Outliers)自動處理",
        "跨系統異構數據格式標準化對齊", "大規模Excel/CSV表格自動化公式運算",
        "數據透視表(Pivot Table)自動構建與多維度切片", "樣本數據分層抽樣與代表性檢驗",
        "數據欄位血緣追蹤與計算邏輯解釋", "合成測試數據(Synthetic Data)高擬真生成",
        "描述性統計指標(均值、中位數、標準差等)自動計算", "相關性分析與關鍵驅動因子識別",
        "A/B測試統計顯著性檢定與結果歸因", "用戶留存率與同期群分析(Cohort Analysis)",
        "銷售漏斗(Funnel)各階段轉化率與流失點診斷", "RFM客戶價值分群模型自動運算",
        "購物籃關聯規則分析(找出常被一起購買的商品)", "動態商業圖表(折線、熱力、散點、桑基圖)自動生成",
        "數據異常波動自動預警與根因推演(Root Cause Analysis)", "商業指標(KPI)達成率追蹤與差距分析",
        "財務三表(損益、資產負債、現金流)自動解讀", "企業償債能力、營運能力與盈利能力比率計算",
        "預算執行偏差分析與成本結構拆解", "投資專案NPV(淨現值)與IRR(內部報酬率)測算",
        "動態定價策略與價格彈性模擬", "供應鏈庫存安全水位與補貨週期測算",
        "金融市場時間序列趨勢與波動率分析", "投資組合風險敞口與夏普比率評估",
        "稅務合規性初步檢核與稅負測算", "商業模式單位經濟效益(Unit Economics)拆解",
        "基於歷史數據的未來銷售額與需求預測", "客戶流失預警(Churn Prediction)機率評估",
        "多維度決策矩陣(Weighted Decision Matrix)打分與排序", "商業情景模擬(What-If Analysis:樂觀/基準/悲觀推演)",
        "資源分配最佳化與線性規劃求解", "專案排程關鍵路徑(CPM)與瓶頸識別",
        "競品優勢SWOT與波特五力模型深度推演", "定價與促銷活動ROI事前預估",
        "選址與市場進入可行性多維度評估", "供應商綜合績效評分與淘汰建議",
        "商業合約法律風險與不平等條款審查", "企業內部控制與合規性漏洞掃描",
        "反洗錢(AML)與異常交易行為模式識別", "大規模客戶評論與輿情情感極性量化分析",
        "客戶痛點聚類與需求優先級排序", "複雜商業問題的第一性原理拆解",
        "邏輯謬誤與論證漏洞自動識別(紅隊思維審查)", "複雜任務目標拆解與階段性行動計畫(Task Planning)",
        "數學公式推導與符號運算驗證", "因果推斷(Causal Inference)與干擾變因排除"
    ],
    SkillCategory.CONTENT_CREATION: [
        "品牌核心價值主張(Value Proposition)提煉", "品牌語氣與寫作風格(Brand Voice)克隆",
        "行業深度白皮書與研究報告撰寫", "SEO搜尋引擎優化長文生成(含關鍵字佈局)",
        "企業創辦人與高管思想領袖(Thought Leadership)專欄代筆", "客戶成功案例(Case Study)故事化撰寫",
        "企業年度報告與股東信潤飾撰寫", "產品發布會主題演講稿(Keynote Speech)撰寫",
        "品牌故事與企業文化手冊編寫", "深度人物專訪稿整理與文學化潤飾",
        "高點擊率(CTR)文章與影片標題批量生成", "Facebook/Instagram高互動貼文與Hashtag組合",
        "LinkedIn專業商業人脈貼文撰寫", "小紅書/Threads爆款種草筆記與網感文案",
        "X(Twitter)高傳播力推文串(Thread)拆解撰寫", "一魚多吃:單一核心長文自動改寫為全平台矩陣貼文",
        "社群留言區高情商互動回覆與引導轉化話術", "KOL/網紅合作邀約信與推廣Brief撰寫",
        "社群抽獎與裂變活動規則與宣傳文案", "節日與熱點借勢行銷(Newsjacking)快速文案生成",
        "Google/Meta搜尋與資訊流廣告多版本測試文案", "高轉化率到達頁面(Landing Page)架構與銷售文案",
        "亞馬遜/蝦皮/淘寶產品詳情頁(Listing)賣點提煉", "電商商品標題關鍵字權重優化",
        "購物車棄單挽回(Abandoned Cart)促銷簡訊與郵件", "直播帶貨逐字稿腳本(含痛點切入、逼單話術與互動節奏)",
        "產品包裝盒文案與使用說明書精簡撰寫", "針對不同受眾畫像(Persona)的差異化銷售說辭",
        "競品對比頁面(Comparison Page)說服性文案", "客戶評價(Reviews)自動回覆與負評安撫話術",
        "TikTok/Reels/Shorts短影音黃金前三秒吸睛腳本", "商業廣告片與品牌微電影分鏡腳本(含畫面與旁白)",
        "YouTube知識型長影片大綱與口播稿撰寫", "Podcast播客節目訪談提綱與開場白設計",
        "高開信率B2B陌生開發信(Cold Email)個人化撰寫", "商業合作提案書(Proposal)與企劃案架構撰寫",
        "募資簡報(Pitch Deck)每頁核心文案與投資人問答準備", "媒體新聞稿(Press Release)標準格式撰寫",
        "危機公關官方聲明稿與媒體應對Q&A", "內部全員信與組織變革溝通公告撰寫",
        "線上課程大綱設計與逐字講義編寫", "企業新人培訓SOP標準作業程序手冊撰寫",
        "員工考核與面試結構化題庫生成", "客服標準話術庫(FAQ)自動擴寫與場景覆蓋",
        "銷售團隊異議處理(Objection Handling)實戰手冊", "文案去AI味與人類自然語感深度潤飾",
        "文本可讀性評分與複雜長句拆解簡化", "跨文化行銷文案創意改寫(Transcreation,非字面直譯)",
        "文本情感色彩調整(如:將強硬語氣轉為委婉專業)", "文案合規性與廣告法違禁詞自動審查替換"
    ],
    SkillCategory.VISUAL_DESIGN: [
        "商業級產品攝影場景合成(保留商品細節替換背景光影)", "虛擬模特兒(AI Model)服飾穿搭與試穿效果生成",
        "品牌視覺一致性IP角色與吉祥物多姿勢生成", "商業插畫、扁平風與3D渲染風格海報生成",
        "Logo標誌概念草圖與品牌VI視覺系統發想", "包裝設計結構展開圖與3D立體效果預覽",
        "室內設計與建築空間效果圖快速渲染", "書籍封面、專輯封面與活動主視覺(Key Visual)生成",
        "資訊圖表(Infographic)視覺元素與圖標(Icon)批量生成", "藝術字體設計與創意字效生成",
        "髮絲級複雜邊緣智能去背與透明底圖輸出", "圖像無損高清放大(Upscaling)與細節重繪",
        "老照片修復、去噪點與黑白照片智能上色", "畫面多餘路人、雜物或浮水印智能擦除(Inpainting)",
        "圖像邊界智能擴展(Outpainting,如橫圖轉直圖)", "局部重繪與元素替換(如更換人物服裝、髮型或表情)",
        "圖像光影重打光(Relighting)與色調統一批次處理", "照片風格遷移(如將實拍照片轉為油畫、動漫或賽博龐克風)",
        "電商主圖與社群圖文自動排版套版生成", "圖片OCR文字識別與原位置無縫翻譯替換",
        "網頁與手機App UI介面高保真視覺原型生成", "手繪線框草圖(Wireframe)自動轉化為標準前端設計稿",
        "設計系統(Design System)色彩規範與元件庫自動生成", "單張2D產品照片轉3D網格模型(Mesh)",
        "3D模型PBR材質貼圖自動生成", "3D角色自動綁定骨骼(Auto-Rigging)與基礎動作生成",
        "遊戲與虛擬展廳3D場景環境資產生成", "向量圖形(SVG)自動繪製與路徑優化",
        "簡報(PPT/Keynote)視覺版面自動美化與配圖", "數據視覺化動態大屏(Dashboard)UI設計",
        "純文字提示詞生成高畫質創意短片(Text-to-Video)", "靜態產品海報或插畫轉動態宣傳影片(Image-to-Video)",
        "數字人(AI Avatar)擬真口播影片生成", "影片多語言配音與嘴型精準同步(Lip-Sync)",
        "長影片自動識別高光時刻並剪輯為短影音精華", "影片智能去背景(無綠幕人物摳像)",
        "影片畫質增強、補幀(60fps)與防手震穩定處理", "影片風格化轉繪(Video-to-Video,如真人影片轉動畫)",
        "影片自動生成多語言雙語字幕與動態花字", "影片內容自動審查(識別違規畫面與版權片段)",
        "多語種高擬真語音合成(TTS,含情緒、停頓與呼吸聲)", "特定人物音色極速克隆(Voice Cloning)",
        "語音轉文字(STT)高精度轉錄(支援方言與專業術語)", "錄音檔人聲增強、環境噪音與回音消除",
        "歌曲人聲與伴奏音軌無損分離", "無版權商業背景音樂(BGM)依氛圍與時長定制生成",
        "完整人聲演唱歌曲生成(含作詞、作曲、編曲與演唱)", "影視與遊戲專屬音效(Foley SFX,如腳步聲、爆炸聲)生成",
        "語音情感與語調實時轉換(Voice-to-Voice)", "Podcast與有聲書自動母帶處理(Mastering)與音量標準化"
    ],
    SkillCategory.PROGRAMMING: [
        "跨程式語言(Python/JS/Go/Rust/C++)功能模組自動生成", "自然語言需求直接生成完整全端應用程式原型",
        "遺留舊程式碼(Legacy Code)重構與效能優化", "程式語言跨平台自動遷移(如Java轉Go、Python 2轉3)",
        "程式碼架構設計與設計模式(Design Patterns)建議", "演算法時間與空間複雜度分析與極致優化",
        "自動化程式碼審查(Code Review)與規範檢查", "程式碼安全漏洞掃描(識別SQL注入、XSS、權限越權)",
        "第三方開源套件依賴衝突檢測與安全版本升級建議", "複雜正規表達式(Regex)與SQL查詢語句自動生成與解釋",
        "自動化單元測試(Unit Test)與邊界條件案例生成", "端到端(E2E)自動化測試腳本(Playwright/Cypress)編寫",
        "系統錯誤日誌(Error Logs)自動分析與根因定位", "程式Bug自動修復補丁(Patch)生成與驗證",
        "伺服器效能瓶頸診斷與記憶體洩漏排查", "Docker容器化配置(Dockerfile)與Kubernetes編排檔生成",
        "CI/CD自動化部署流水線(GitHub Actions/GitLab CI)配置", "雲端架構(AWS/GCP/Azure)基礎設施即程式碼(Terraform)生成",
        "終端機指令(Bash/PowerShell)自然語言轉換與執行", "系統異常自動自癒腳本觸發(如服務崩潰自動重啟並回報)",
        "RESTful/GraphQL API介面快速開發與模擬(Mock)", "跨系統異構API數據格式自動轉換中間件編寫",
        "Webhook事件監聽與即時自動化觸發器配置", "資料庫Schema設計、ER圖生成與索引優化建議",
        "資料庫遷移腳本(Migration)與無損備份自動化", "OAuth2/JWT身份驗證與權限控制模組生成",
        "企業內部系統(CRM/ERP/Slack/飛書/Teams)消息機器人串接", "雲端檔案系統(Google Drive/S3/Notion)自動同步與權限管理",
        "支付網關(Stripe/PayPal)與訂閱計費邏輯整合", "郵件伺服器自動收發、解析與觸發工作流",
        "反爬蟲機制應對與動態渲染網頁(SPA)數據抓取", "模擬人類操作行為的無頭瀏覽器(Headless Browser)自動化",
        "無API老舊系統的網頁表單自動填寫與提交(RPA)", "跨網站批量帳號登入、狀態維持與資料匯出",
        "自動化檔案下載、重新命名、壓縮與歸檔腳本", "桌面端GUI介面元素識別與自動化點擊操作",
        "驗證碼(CAPTCHA)智能識別與自動化處理對接", "分散式爬蟲排程與代理IP池自動輪換管理",
        "電商平台庫存與價格自動化採集管道", "政府與企業招投標網站自動化監控與標書下載",
        "程式碼庫自動生成詳細技術文件與架構說明書", "OpenAPI/Swagger介面文檔自動生成",
        "Git Commit提交訊息與版本發布日誌(Release Notes)自動生成", "提示詞工程(Prompt Engineering)自動優化與測試評估",
        "大型語言模型(LLM)輸出格式強制校驗(JSON Schema Guardrails)", "模型路由調度(依任務難度自動分配高階或低成本模型以節省算力)",
        "向量資料庫(Vector DB)索引構建、分塊(Chunking)策略優化", "工具調用(Function Calling)介面封裝與安全沙盒(Sandbox)執行",
        "人機協同審批節點(Human-in-the-Loop)卡點控制與通知", "多代理協作總調度(Multi-Agent Orchestration):負責任務拆解、狀態追蹤、呼叫上述249項技能並驗收最終交付成果"
    ]
}


@dataclass
class SkillGene:
    """Skill gene definition."""
    gene_key: GeneKey
    name: str
    description: str
    category: SkillCategory
    index: int
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    status: str = "available"  # available, downloading, researching, testing, promoted, deprecated


class SkillRegistry:
    """Registry for all 250 elite skills."""
    
    def __init__(self):
        self._genes: Dict[str, SkillGene] = {}
        self._skill_specs: Dict[str, SkillSpec] = {}
        self._initialize_genes()
    
    def _initialize_genes(self) -> None:
        """Initialize all 250 elite skill genes."""
        for category, skill_names in ELITE_SKILL_GENES.items():
            for index, name in enumerate(skill_names, 1):
                gene_key = GeneKey(category, index)
                
                # Create skill gene
                gene = SkillGene(
                    gene_key=gene_key,
                    name=name,
                    description=f"Elite skill: {name}",
                    category=category,
                    index=index,
                    input_schema=self._get_input_schema(category, name),
                    output_schema=self._get_output_schema(category, name),
                    parameters=self._get_parameters(category, name),
                    resource_requirements=self._get_resource_requirements(category, name),
                    tags=[category.value]
                )
                
                self._genes[str(gene_key)] = gene
        
        logger.info(f"Initialized {len(self._genes)} elite skill genes")
    
    def _get_input_schema(self, category: SkillCategory, name: str) -> Dict[str, Any]:
        """Get input schema for skill."""
        base_schemas = {
            SkillCategory.INFORMATION_RETRIEVAL: {"query": "string", "sources": "array", "limit": "integer"},
            SkillCategory.LOGICAL_REASONING: {"data": "object", "parameters": "object"},
            SkillCategory.CONTENT_CREATION: {"topic": "string", "style": "string", "length": "integer"},
            SkillCategory.VISUAL_DESIGN: {"prompt": "string", "style": "string", "dimensions": "object"},
            SkillCategory.PROGRAMMING: {"requirements": "string", "language": "string", "framework": "string"}
        }
        return base_schemas.get(category, {"input": "any"})
    
    def _get_output_schema(self, category: SkillCategory, name: str) -> Dict[str, Any]:
        """Get output schema for skill."""
        base_schemas = {
            SkillCategory.INFORMATION_RETRIEVAL: {"results": "array", "sources": "array", "confidence": "number"},
            SkillCategory.LOGICAL_REASONING: {"result": "object", "analysis": "string", "confidence": "number"},
            SkillCategory.CONTENT_CREATION: {"content": "string", "metadata": "object"},
            SkillCategory.VISUAL_DESIGN: {"assets": "array", "metadata": "object"},
            SkillCategory.PROGRAMMING: {"code": "string", "tests": "string", "documentation": "string"}
        }
        return base_schemas.get(category, {"output": "any"})
    
    def _get_parameters(self, category: SkillCategory, name: str) -> Dict[str, Any]:
        """Get default parameters for skill."""
        return {
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60
        }
    
    def _get_resource_requirements(self, category: SkillCategory, name: str) -> Dict[str, Any]:
        """Get resource requirements for skill."""
        return {
            "cpu_cores": 1,
            "memory_mb": 512,
            "gpu_memory_mb": 0,
            "estimated_duration_seconds": 30
        }
    
    def get_gene(self, gene_key: GeneKey) -> Optional[SkillGene]:
        return self._genes.get(str(gene_key))
    
    def get_all_genes(self) -> List[SkillGene]:
        return list(self._genes.values())
    
    def get_genes_by_category(self, category: SkillCategory) -> List[SkillGene]:
        return [g for g in self._genes.values() if g.category == category]
    
    def get_gene_keys(self) -> List[GeneKey]:
        return [g.gene_key for g in self._genes.values()]
    
    def register_skill_spec(self, gene_key: GeneKey, spec: SkillSpec) -> None:
        self._skill_specs[str(gene_key)] = spec
    
    def get_skill_spec(self, gene_key: GeneKey) -> Optional[SkillSpec]:
        return self._skill_specs.get(str(gene_key))
    
    def get_status(self, gene_key: GeneKey) -> str:
        gene = self._genes.get(str(gene_key))
        return gene.status if gene else "unknown"
    
    def update_status(self, gene_key: GeneKey, status: str) -> None:
        gene = self._genes.get(str(gene_key))
        if gene:
            gene.status = status


class SkillExecutor:
    """Executes skills using LLM, sandbox, or external tools."""
    
    def __init__(
        self,
        registry: SkillRegistry,
        llm_provider: LLMProviderPort,
        sandbox: SandboxPort,
        open_source: OpenSourceSearchPort
    ):
        self._registry = registry
        self._llm = llm_provider
        self._sandbox = sandbox
        self._open_source = open_source
    
    @traced("skill_executor.execute")
    async def execute(
        self,
        gene_key: GeneKey,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Result[Dict[str, Any]]:
        """Execute a skill."""
        gene = self._registry.get_gene(gene_key)
        if not gene:
            return failure(SkillError(f"Skill {gene_key} not found"))
        
        # Check if we have a registered spec
        spec = self._registry.get_skill_spec(gene_key)
        
        if spec and spec.source == "builtin":
            return await self._execute_builtin(gene, input_data, context)
        elif spec and spec.source in ["downloaded", "researched"]:
            return await self._execute_custom(spec, input_data, context)
        else:
            # Try to acquire skill
            return failure(SkillError(f"Skill {gene_key} not available. Use acquire_skill first."))
    
    async def _execute_builtin(
        self,
        gene: SkillGene,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """Execute builtin skill via LLM."""
        # Construct prompt based on skill
        prompt = self._build_prompt(gene, input_data, context)
        
        system_prompt = f"You are an expert in {gene.name}. Execute the task precisely."
        
        result = await self._llm.complete(prompt, system_prompt=system_prompt)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Parse result
        output = result.unwrap()
        
        return success({
            "success": True,
            "skill": str(gene.gene_key),
            "output": output,
            "parsed": self._parse_output(gene, output)
        })
    
    async def _execute_custom(
        self,
        spec: SkillSpec,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """Execute custom skill (downloaded or researched)."""
        # Would execute custom code in sandbox
        if spec.code:
            sandbox_result = await self._sandbox.execute(
                code=spec.code,
                language="python",
                timeout=spec.resource_requirements.get("timeout", 300)
            )
            if sandbox_result.is_failure():
                return failure(sandbox_result.unwrap_err())
            
            return success({
                "success": True,
                "skill": str(spec.gene_key),
                "output": sandbox_result.unwrap()
            })
        
        return failure(SkillError(f"No executable code for skill {spec.gene_key}"))
    
    def _build_prompt(self, gene: SkillGene, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Build prompt for skill execution."""
        return f"""
        Skill: {gene.name}
        Category: {gene.category.value}
        
        Input: {json.dumps(input_data, ensure_ascii=False)}
        Context: {json.dumps(context, ensure_ascii=False) if context else 'None'}
        
        Execute this skill and provide output in the expected format.
        """
    
    def _parse_output(self, gene: SkillGene, output: str) -> Dict[str, Any]:
        """Parse skill output."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}


class SkillAcquisitionManager:
    """Manages skill acquisition from open source or research."""
    
    def __init__(
        self,
        registry: SkillRegistry,
        executor: SkillExecutor,
        open_source: OpenSourceSearchPort,
        llm_provider: LLMProviderPort,
        sandbox: SandboxPort,
        audit_brain: 'AuditBrain'
    ):
        self._registry = registry
        self._executor = executor
        self._open_source = open_source
        self._llm = llm_provider
        self._sandbox = sandbox
        self._audit_brain = audit_brain
    
    @traced("skill_acquisition.acquire")
    async def acquire_skill(self, gene_key: GeneKey) -> Result[SkillSpec]:
        """Acquire skill - try open source first, then research."""
        # Check if already available
        spec = self._registry.get_skill_spec(gene_key)
        if spec:
            return success(spec)
        
        gene = self._registry.get_gene(gene_key)
        if not gene:
            return failure(SkillError(f"Gene {gene_key} not in registry"))
        
        self._registry.update_status(gene_key, "downloading")
        
        # Step 1: Search open source
        search_result = await self._search_open_source(gene)
        if search_result.is_success():
            spec = self._create_spec_from_search(gene, search_result.unwrap())
            if spec:
                self._registry.register_skill_spec(gene_key, spec)
                self._registry.update_status(gene_key, "available")
                return success(spec)
        
        # Step 2: Research module creates skill
        self._registry.update_status(gene_key, "researching")
        
        research_result = await self._research_skill(gene)
        if research_result.is_success():
            spec = research_result.unwrap()
            self._registry.register_skill_spec(gene_key, spec)
            self._registry.update_status(gene_key, "testing")
            return success(spec)
        
        self._registry.update_status(gene_key, "failed")
        return failure(SkillError(f"Failed to acquire skill {gene_key}"))
    
    async def _search_open_source(self, gene: SkillGene) -> Result[List[Dict[str, Any]]]:
        """Search open source for skill implementation."""
        # Search code repositories
        code_result = await self._open_source.search_code(
            query=gene.name,
            languages=["python", "javascript", "rust", "go"],
            limit=10
        )
        
        if code_result.is_success() and code_result.unwrap():
            return code_result
        
        # Search papers for algorithms
        paper_result = await self._open_source.search_papers(
            query=gene.name,
            sources=["arxiv", "semantic_scholar", "papers_with_code"],
            limit=10
        )
        
        return paper_result
    
    def _create_spec_from_search(self, gene: SkillGene, results: List[Dict]) -> Optional[SkillSpec]:
        """Create skill spec from search results."""
        if not results:
            return None
        
        # Use first result
        result = results[0]
        
        return SkillSpec(
            gene_key=gene.gene_key,
            name=gene.name,
            description=gene.description,
            input_schema=gene.input_schema,
            output_schema=gene.output_schema,
            parameters=gene.parameters,
            source="downloaded",
            version="1.0.0",
            tags=gene.tags,
            resource_requirements=gene.resource_requirements
        )
    
    async def _research_skill(self, gene: SkillGene) -> Result[SkillSpec]:
        """Use research module to create skill."""
        # This would invoke the research module
        # For now, create a template spec
        
        prompt = f"""
        Create a complete implementation for this skill:
        
        Skill: {gene.name}
        Category: {gene.category.value}
        Description: {gene.description}
        Input Schema: {gene.input_schema}
        Output Schema: {gene.output_schema}
        
        Provide:
        1. Python implementation
        2. Unit tests
        3. Documentation
        4. Resource requirements
        """
        
        result = await self._llm.complete(prompt, system_prompt="You are an expert software engineer creating production-ready skill implementations.")
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Parse and create spec
        spec = SkillSpec(
            gene_key=gene.gene_key,
            name=gene.name,
            description=gene.description,
            input_schema=gene.input_schema,
            output_schema=gene.output_schema,
            parameters=gene.parameters,
            source="researched",
            version="1.0.0",
            tags=gene.tags,
            resource_requirements=gene.resource_requirements
        )
        
        return success(spec)
    
    async def promote_skill(self, gene_key: GeneKey, benchmark_results: Dict[str, float]) -> Result[None]:
        """Promote skill after successful benchmarking."""
        # Audit the skill
        audit_result = await self._audit_brain.process({
            "action": "approve_result",
            "result": benchmark_results,
            "criteria": {
                "accuracy": 0.8,
                "performance": 0.7,
                "reliability": 0.8
            }
        })
        
        if audit_result.is_failure() or not audit_result.unwrap().get("approved"):
            return failure(SkillError("Skill failed audit"))
        
        self._registry.update_status(gene_key, "promoted")
        return success(None)


class SkillValidator:
    """Validates skills in sandbox environment."""
    
    def __init__(self, sandbox: SandboxPort, audit_brain: 'AuditBrain'):
        self._sandbox = sandbox
        self._audit_brain = audit_brain
    
    @traced("skill_validator.validate")
    async def validate_skill(
        self,
        spec: SkillSpec,
        test_cases: List[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """Validate skill with test cases in sandbox."""
        if not spec.code:
            return failure(SkillError("No code to validate"))
        
        results = []
        for test_case in test_cases:
            # Run test in sandbox
            test_code = f"""
            {spec.code}
            
            # Test case
            input_data = {test_case.get('input', {})}
            expected = {test_case.get('expected', {})}
            
            result = execute_skill(input_data)
            
            # Validate
            passed = result == expected
            print(f"TEST_RESULT: {{'passed': passed, 'result': result, 'expected': expected}}")
            """
            
            sandbox_result = await self._sandbox.execute(
                code=test_code,
                language="python",
                timeout=60
            )
            
            if sandbox_result.is_success():
                output = sandbox_result.unwrap()
                # Parse test result
                results.append({
                    "test_case": test_case,
                    "output": output,
                    "passed": "TEST_RESULT: {'passed': True" in output.get("stdout", "")
                })
            else:
                results.append({
                    "test_case": test_case,
                    "error": str(sandbox_result.unwrap_err()),
                    "passed": False
                })
        
        passed_count = sum(1 for r in results if r.get("passed"))
        total = len(results)
        
        return success({
            "total_tests": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": passed_count / total if total > 0 else 0,
            "results": results
        })
    
    async def security_audit(self, spec: SkillSpec) -> Result[Dict[str, Any]]:
        """Security audit of skill code."""
        if not spec.code:
            return success({"audited": False, "reason": "no_code"})
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "eval(", "exec(", "subprocess", "os.system",
            "__import__", "open(", "file(", "socket",
            "requests.", "urllib", "http.client"
        ]
        
        issues = []
        for pattern in dangerous_patterns:
            if pattern in spec.code:
                issues.append(f"Potentially dangerous pattern: {pattern}")
        
        # Audit with audit brain
        audit_result = await self._audit_brain.process({
            "action": "compliance_check",
            "artifact": {"code": spec.code, "type": "skill"}
        })
        
        return success({
            "audited": True,
            "static_analysis_issues": issues,
            "audit_result": audit_result.unwrap() if audit_result.is_success() else {}
        })