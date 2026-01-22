"""
Structured Diagnostic Report Format:
Diagnosis:
  Malnutrition_Status: <Severe Acute Malnutrition | Moderate Acute Malnutrition | Chronic Malnutrition | Overweight | No Malnutrition>
  Risk_Level: <High | Moderate | Low>

Type:
  Classification: <Marasmus | Kwashiorkor | Marasmic-Kwashiorkor | None>

Clinical_Summary:
  Description: |
    <Concise clinical interpretation using provided data>
  Key_Findings:
    - <finding 1>
    - <finding 2>

WHO_Criteria_Triggered:
  - <criterion 1>
  - <criterion 2>

Micronutrient_Deficiency_Suspicions:
  - Nutrient: <name>
    Severity: <Mild | Moderate | Severe>
    Clinical_Signs: <signs>

Recommended_Management:
  Immediate_Actions:
    - <action 1>
    - <action 2>
  Feeding_Protocol:
    - <RUTF / F-75 / F-100 / Balanced diet>
  Referral:
    - <Outpatient | Inpatient | Tertiary facility>
  Follow_Up:
    - <timeline and monitoring>

Alerts:
  - <Red flags requiring urgent care>

Confidence_Level:
  Assessment_Confidence: <High | Moderate | Low>
  Notes: <missing data or assumptions>


"""

from typing import Optional
from pydantic import BaseModel, Field

class DiagnosisSection(BaseModel):
    Malnutrition_Status: str = Field(description="Malnutrition status")
    Risk_Level: str = Field(description="Risk level")

class TypeSection(BaseModel):
    Classification: str = Field(description="Type classification")  

class ClinicalSummarySection(BaseModel):
    Description: str = Field(description="Concise clinical interpretation")
    Key_Findings: list[str] = Field(description="Key findings list")

class MicronutrientDeficiencySuspicion(BaseModel):
    Nutrient: str = Field(description="Nutrient name")
    Severity: str = Field(description="Severity level")
    Clinical_Signs: str = Field(description="Clinical signs")

class RecommendedManagementSection(BaseModel):
    Immediate_Actions: list[str] = Field(description="Immediate actions list")
    Feeding_Protocol: list[str] = Field(description="Feeding protocol options")
    Referral: list[str] = Field(description="Referral options")
    Follow_Up: list[str] = Field(description="Follow-up timeline and monitoring")
    Alerts: list[str] = Field(description="Red flags requiring urgent care")

class ConfidenceLevelSection(BaseModel):
    Assessment_Confidence: str = Field(description="Assessment confidence level")
    Notes: str = Field(description="Additional notes on missing data or assumptions")

class DiagnosticReport(BaseModel):
    """Structured diagnostic report for malnutrition assessment."""
    Diagnosis: DiagnosisSection = Field(description="Diagnosis section")
    Type: TypeSection = Field(description="Type section")
    Clinical_Summary: ClinicalSummarySection = Field(description="Clinical summary section")
    WHO_Criteria_Triggered: list[str] = Field(description="WHO criteria triggered")
    Micronutrient_Deficiency_Suspicions: list[MicronutrientDeficiencySuspicion] = Field(description="Micronutrient deficiency suspicions")
    Recommended_Management: Optional[RecommendedManagementSection] = Field(description="Recommended management section")
    Confidence_Level: ConfidenceLevelSection = Field(description="Confidence level section")
    
