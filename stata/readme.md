# **Carbon Emissions Trading Policy and Corporate Investment Efficiency**

## **Project Overview**

This project studies how China’s carbon emissions trading policy affects the investment efficiency of A-share listed firms.
The analysis uses firm-level panel data from 2008 to 2022 and applies a Difference-in-Differences (DID) model by comparing firms inside and outside the pilot regions.
The goal is to evaluate whether the policy reduces inefficient investment and improves capital allocation.


## **What This Project Does**

### **Data Processing**

* Collects data for all A-share firms from 2008–2022.
* Removes financial firms, ST/*ST/PT firms, and firms with missing key indicators.
* Merges annual financial statements, cash flow data, and ownership data from CSMAR and WIND.
* Winsorizes continuous variables at the 1% and 99% levels to mitigate extreme values.
* Constructs policy variables:

  * `Treat = 1` for firms in Beijing, Tianjin, Shanghai, Chongqing, Hubei, Guangdong, Shenzhen
  * `Post = 1` for years ≥ 2011
  * `TreatPost = Treat × Post`

### **Model and Identification**

The DID model:

```
Y_it = α + β (Treat_i × Post_t) + γ X_it + θ_t + μ_j + ε_it
```

Where:

* `Y_it` includes:

  * inefficient investment (`IneInvest`)
  * over-investment (`OverInv`)
  * under-investment (`WeakInv`)
* `X_it` contains firm controls: leverage, firm age, asset size, growth opportunity, ROA, CFO, ownership structure.
* `θ_t`: year fixed effects
* `μ_j`: industry fixed effects
* Standard errors are robust.

### **Measurement of Investment Inefficiency**

Investment efficiency is measured using the Richardson (2006) model:

* Compute expected investment value.
* Obtain residuals from the expected–actual investment equation.

  * `|residual| = Inefficient investment`
  * `residual > 0 → OverInv`
  * `residual < 0 → WeakInv = abs(residual)`


## **Technical Workflow**

### **Step 1: Data Preparation**

* Format financial indicators and convert selected variables (such as firm size and age) into log values.
* Create an index for industry classification.
* Combine multiple datasets into a longitudinal panel.
* Export descriptive statistics and correlation matrices for validation.

### **Step 2: DID Estimation**

* Estimate policy impact on each investment-efficiency variable.
* Include year and industry fixed effects to control for macroeconomic fluctuations and sector-level patterns.
* Check coefficient signs and compare magnitudes across models.

### **Step 3: Robustness Tests**

#### **Event-Study / Parallel Trend Check**

Uses year-relative dummy variables:


Before3 Before2 Before1 Current After1 After2 After3


to observe dynamic policy effects across time.

#### **Placebo Test**

* Randomly assign firms into “treatment” groups for 500 simulations.
* Re-estimate DID for each simulated dataset.
* Collect simulated `β` coefficients and compare with the true estimate.
* Inspect distribution plots and p-value scatter results.

#### **Alternative Policy Year**

* Re-define policy start year as 2013 (actual market launch).
* Re-run all baseline regressions to verify stability.

### **Step 4: Output Reporting**

* Export regression tables using DOCX format.
* Create plots for:

  * dynamic treatment effects
  * placebo coefficient distribution
* Save regression models for cross-table comparison.



## **Key Findings**

### **1. Firms Improve Investment Efficiency After Policy Begins**

The DID coefficient for `TreatPost` is negative and significant.
Firms in pilot regions reduce inefficient investment more than those in non-pilot regions.

### **2. Over-Investment and Under-Investment Both Decline**

* Excess expansion becomes less common.
* Firms with under-investment gain better financing channels and confidence to invest.

### **3. Results Stay Consistent Across Validation Tests**

* Parallel trends hold before policy intervention.
* Placebo outcomes concentrate near zero, unlike the real effect.
* Changing policy year yields similar coefficient patterns.


## **Repository Structure**

```
data/           # Raw and cleaned datasets
code/           # Stata scripts for cleaning, DID models, and robustness checks
results/        # Exported tables, figures, and model outputs
paper/          # Full research paper and reference materials
README.md




## **Methods and Tools**

* **Stata**

  * DID estimation
  * Fixed-effects regression
  * winsor2, reghdfe, coefplot, reg2docx
* **CSMAR / WIND** for firm-level data
* **DOCX** export for publication-ready tables
* **Panel data methods** with year + industry fixed effects



## **Summary**

This project provides a clear workflow for evaluating the impact of environmental regulation on corporate investment behavior.
Empirical evidence suggests that China’s carbon emissions trading policy helps firms reduce inefficient investment and allocate capital more effectively.



