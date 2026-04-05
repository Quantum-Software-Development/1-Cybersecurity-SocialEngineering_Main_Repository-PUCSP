
# **EXPLORATORY PROJECT REPORT**

**THEME**: Analysis of AI Incidents in Financial Services: Algorithmic Bias, Operational Risk, and Governance Responses in Banks and Fintechs

---

## **1. Project Objective**

### **1.1 General Objective**
This project aims to **evaluate, using structured data from AI system incidents in the financial sector**, whether there are **systematic patterns of algorithmic bias, operational risk, and governance failures** in AI applications within **banks and fintechs**.

I will use **relational databases, statistical analyses, and ML models** to generate evidence supporting **decision-making in AI ethics policies, risk management, and governance** in the financial market.

### **1.2 Specific Objectives**
1. **Identify and quantify** documented AI incidents in finance, focusing on credit scoring, fraud detection, algorithmic trading, and banking automation
2. **Analyze severity** of impacts caused (financial losses, reputational damage, customer exclusion) and their distribution across client segments
3. **Evaluate governance responses** (regulatory investigations, fines, policy changes) implemented following reported incidents
4. **Build simple predictive models** to estimate risk and severity of future incidents in financial institutions
5. **Develop a RESTful API** exposing data and analyses accessibly for risk managers, regulators, and decision-makers

### **1.3 Research Questions**
1. Is there **concentration of AI incidents in specific financial applications** (credit, fraud, trading)?
2. Do **algorithmic bias incidents in credit scoring disproportionately affect** certain client segments (small businesses, minorities, specific regions)?
3. Do **high-severity incidents** (significant financial losses, systemic failures) result in **proportional regulatory actions**?
4. Is it possible to **identify temporal trends** indicating increase/decrease in incidents over time, correlated with regulatory evolution?

---

## **2. Data Used and Preparation**

### **2.1 Data Source**
Data comes from the **Artificial Intelligence Incident Database (AIID)**, a repository gathering documented AI incidents that caused or nearly caused real-world harm.

**Access methods**:
- **Kaggle tabular dataset**: AI Incident Database
- **Direct download** from AIID official site

**AIID sources**: Financial news (Bloomberg, Financial Times, Reuters), regulatory reports, court decisions, academic articles – **already structured as tables** with identifiers, dates, locations, summaries, application categories, and damage types.

### **2.2 Preparation Process**
1. **Load CSV** in Python/Google Colab using pandas library
2. **Initial inspection**: columns, data types, sample rows
3. **Select relevant raw columns** for project scope
4. **Cleaning**: date standardization, country/sector normalization, missing value treatment
5. **Thematic filtering + enrichment**: Focus on financial sector incidents
6. **Store structured variables** in **SQLite relational database**

### **2.3 Key Columns Selected**
```

incident_id: unique identifier
title: headline/title of incident
summary: textual summary of occurrence
occurred_date: occurrence date
country: country/region where incident occurred
sector: sector/application (focus: finance)
technology: AI technology name/type involved
harm_type: main damage type (fairness, safety, financial loss, etc.)
severity: severity level when available

```

---

## **3. Study Variables**

### **3.1 Raw Variables Extracted from Dataset**
```

incident_id (numeric): unique incident identifier
title (text): headline describing incident
summary (text): descriptive summary of what happened
occurred_date (date): date of incident occurrence
country (categorical): country where incident occurred
sector (categorical): application sector (finance focus)
technology (text): type of AI technology used
harm_type (categorical): main damage type
severity (categorical/numeric): incident severity level

```

### **3.2 Derived Variables Created for Analysis**

#### **3.2.1 Financial Application Type** (`application_type` - categorical)
```

credit_scoring: credit evaluation and loan approval systems
fraud_detection: fraud detection and money laundering (AML)
algorithmic_trading: algorithmic trading and high-frequency trading
robo_advisor: automated investment advisory
risk_assessment: portfolio risk assessment and automated underwriting
customer_service: chatbots and customer service automation

```
**Why it matters**: Identifies which AI applications in finance present highest incident/risk concentrations.

#### **3.2.2 Affected Client Segment** (`customer_segment` - categorical)
```

retail: individual retail customers
sme: small and medium enterprises
corporate: large corporations
underserved: under-served groups (minorities, specific regions)
general: systemic/general impact

```
**Why it matters**: Essential to assess if algorithmic bias disproportionately affects vulnerable groups.

#### **3.2.3 Incident Type** (`incident_type` - categorical)
```

algorithmic_bias: systematic bias in automated decisions (e.g., credit discrimination)
operational_failure: technical failure causing losses/service interruption
market_disruption: algorithmic trading error causing volatility/flash crash
data_breach: data leak due to AI system failure
regulatory_violation: automated system regulatory non-compliance

```
**Why it matters**: Organizes incidents into recognized operational risk categories in financial sector.

#### **3.2.4 Impact Severity** (`severity_level` - ordinal)
```

Critical: >\$10M losses, systemic market failure, multiple regulatory violations
High: \$1M-\$10M losses, significant reputational impact, formal regulatory investigation
Medium: \$100K-\$1M losses, customer complaints, mandatory internal review
Low: <\$100K losses, limited impact, immediate correction without external consequences

```
**Additional**: `financial_loss_amount` (numeric): estimated direct financial loss in USD when reported.

#### **3.2.5 Governance & Regulatory Responses** (binary 0/1 variables)
```

regulatory_investigation: formal investigation by regulatory authority
fine_imposed: financial fine/sanction applied
policy_change: declared internal policy/procedure change
system_suspended: temporary/permanent AI system suspension
third_party_audit: independent audit requested/performed

```
**Why they matter**: Evaluate governance effectiveness and regulatory response proportionality to risk.

---

## **4. Hypotheses and Statistical Analysis**

### **4.1 Hypothesis 1: Incident Concentration by Application Type**
**H₀**: Incident distribution uniform across different AI applications in finance  
**H₁**: Certain applications (credit scoring, algorithmic trading) show significantly higher incident concentration  
**Technique**: Chi-square goodness-of-fit test comparing observed vs expected uniform distribution

### **4.2 Hypothesis 2: Bias by Client Segment**
**H₀**: Proportion of bias incidents independent of affected client segment  
**H₁**: Certain segments (SMEs, underserved) disproportionately affected by algorithmic bias in credit decisions  
**Technique**: Chi-square test of independence (algorithmic_bias vs customer_segment)

### **4.3 Hypothesis 3: Severity and Regulatory Response**
**H₀**: Probability of regulatory investigation independent of incident severity  
**H₁**: High-severity incidents (especially significant financial losses) result in higher investigation/fine probability  
**Technique**: Chi-square test (severity_level vs regulatory_investigation), logistic regression

### **4.4 Hypothesis 4: Temporal Trend and Regulatory Evolution**
**H₀**: No significant temporal trend in incident count over years  
**H₁**: Increasing/decreasing trend correlated with regulatory milestones (GDPR, MiCA, AI Act, BACEN regulation)  
**Technique**: Linear regression of annual incident count, Mann-Kendall trend test

---

## **5. ML and Statistical Techniques**

### **5.1 Machine Learning Models**

#### **5.1.1 Severity and Financial Loss Prediction**
**Target variables**: 
- `severity_level` (multiclass classification) 
- `financial_loss_amount` (regression)

**Features**: `application_type`, `customer_segment`, `incident_type`, `country_region`, `year`

**Candidate models**:
1. **Random Forest**: Robust, handles categorical variables, interpretable feature importance
2. **XGBoost**: High performance on tabular data
3. **Logistic Regression**: Interpretability (severity binary)

**Evaluation metrics**:
```

Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Regression: RMSE, MAE, R²

```

---

## **6. Project Structure in Three Phases**

### **Phase 1: Exploratory & Database Construction** (Weeks 1-2)
```

Activities:
✓ Download AIID dataset (Kaggle/AIID site)
✓ Load/inspect in Google Colab (pandas)
✓ Filter financial sector incidents
✓ Create derived variables
✓ Build SQLite database (3 related tables)
Deliverables: Colab notebook, SQLite DB, data dictionary

```

### **Phase 2: Rich Analyses & Hypothesis Tests** (Weeks 3-4)
```

Activities:
✓ Descriptive statistics by variable
✓ Bivariate/multivariate analysis
✓ Execute 4 main hypothesis tests
✓ Interactive dashboard (Streamlit)
Deliverables: Analysis notebook, hypothesis test report, dashboard

```

### **Phase 3: Predictive Models & REST API** (Weeks 5-6)
```

Activities:
✓ ML model development + evaluation
✓ REST API construction (Flask/FastAPI)
✓ 8+ endpoints (incidents, stats, predictions)
✓ Swagger documentation + testing
Deliverables: Trained models, API source code, documentation, demo video

```

---

## **7. Alignment with Course Briefing**

**✅ Subproject 1**: API Consumption/Third-Party Data  
- AIID data via Kaggle/API  
- SQL relational database processing  

**✅ Subproject 2**: REST API Development  
- Custom Flask/FastAPI implementation  
- SQLite 3-table exposure  
- Multiple endpoints (query, analysis, prediction)  

**✅ Data Requirements**:  
- Relational database with 3 related tables  
- Quantitative (losses) + qualitative (types) variables  
- Adequate volume for robust statistical/ML analysis  

---

## **8. Conclusion & Next Steps**

This exploratory report establishes solid foundations for the financial sector-focused pilot project. Selected variables, formulated hypotheses, and planned statistical/ML techniques align with objectives to investigate **algorithmic bias, operational risk, and AI governance** in banks/fintechs.

**Immediate next steps**:
1. Start code implementation in Google Colab
2. Download/load AIID dataset from Kaggle  
3. Initial exploration focused on financial incidents
4. Validate thematic filtering feasibility
5. Create first finance-specific derived variables
6. Establish initial SQLite database schema

---

**Project Title**: Analysis of AI Incidents in Financial Services: Algorithmic Bias, Operational Risk, and Governance Responses in Banks and Fintechs  
**Student**: Fabiana Campaari  
**Institution**: PUC-SP  
**Date**: 2026

