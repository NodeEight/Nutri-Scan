from typing import Optional, List, Union
from pydantic import BaseModel, Field, HttpUrl, field_validator


from schema.diagnosis import DiagnosticReport



json_schema_example = {
    "body_parts": [
        {
            "body_part": "arm",
            "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qycyjl3rlvaz6nfynmwq.jpg",
        },
        {
            "body_part": "leg",
            "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131874/nouritrack/screenings/hqxcwb4ehgkrvks6wfxd.jpg",
        },
        {
            "body_part": "head",
            "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/vma6wutydp0a3cy4gpmc.jpg",
        },
        {
            "body_part": "side",
            "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qymlm1wb3y9flce5uslg.jpg",
        },
    ],
    "vital_measurements": {
        "muac": 12.0,
        "height": 85.0,
        "weight": 10.0,
        "age": 36.0,
        "whz": -2.0,
        "waz": -3.0,
        "haz": -2.0,
    },
    "clinical_findings": {
        "symptoms": ["Swelling (edema) in feet/ankles", "Brittle / Discoloured Hair"],
        "observations": "Additional clinical observations here",
    },
}


BODY_PARTS = ["back", "body", "finger", "head", "leg", "muac", "side"]


class BodyPart(BaseModel):
    body_part: str = Field(
        ..., description=f"Body part name: {', '.join(BODY_PARTS)}", example="arm"
    )
    image_url: str = Field(..., description="Image url of the body part")

    # @field_validator('body_part')
    # def validate_body_part(cls, v):
    #     if v.lower() not in BODY_PARTS:
    #         raise ValueError(f"Invalid body part. Must be one of: {BODY_PARTS}")
    #     return v.lower()


class VitalMeasurements(BaseModel):
    muac: float = Field(
        ...,
        ge=0.0,
        lt=100.0,
        description=" Mid-Upper Arm Circumference measurement in cm",
    )
    height: float = Field(
        ..., ge=0.0, lt=100.0, description=" Height measurement in cm"
    )
    weight: float = Field(
        ..., ge=0.0, lt=100.0, description=" Weight measurement in kg"
    )
    age: float = Field(..., ge=1.0, le=60.0, description=" Age in months")
    hemoglobin: Optional[float] = Field(
        None, ge=0.0, lt=30.0, description=" Hemoglobin level in g/dL"
    )
    whz: Optional[float] = Field(..., description=" Weight-for-height Z-score")
    waz: Optional[float] = Field(..., description=" Weight-for-age Z-score")
    haz: Optional[float] = Field(..., description=" Height-for-age Z-score")


class clinicalFindings(BaseModel):
    symptoms: List[str] = Field(..., description=" List of symptoms")
    observations: Optional[str] = Field(
        ..., description=" Additional clinical observations"
    )


class PredictionRequest(BaseModel):
    screening_id: Optional[str] = Field(
        None, description="Unique identifier for the screening session"
    )
    body_parts: Optional[List[BodyPart]]
    vital_measurements: VitalMeasurements
    clinical_findings: clinicalFindings

    # json_schema_example = json_schema_example


class ModelResponse(BaseModel):
    body_part: str = Field(..., description="Body part name")
    prediction: str = Field(..., description="Prediction result")
    confidence: str = Field(..., description="Confidence score")
    is_nourished: bool = Field(..., description="Is the body part nourished")


class ModelResponseError(BaseModel):
    body_part: str = Field(..., description="Body part name")
    error: str = Field(..., description="Error message")

class DiagnosticResult(BaseModel):
    diagnostic_report: DiagnosticReport = Field(
        ..., description="Structured diagnostic report"
    )
ResponseItem = Union[ModelResponse, DiagnosticResult]
class PredictionAPIResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    results: List[ResponseItem] = Field(
        ..., description="List of prediction results and diagnostic report"
    )

if __name__ == "__main__":
    """
    {
    "screeningId": "69268423ed8293d3b66aeeda",
    "patientInfo": {

        "age": "3 years",
        "gender": "male",
        "region": "ashanti",
        "district": "kumasi"
    },
    "vitalMeasurements": {
        "weight": 20,
        "height": 10,
        "muac": 12,
        "haemoglobin": 12
    },
    "clinicalFindings": {
        "symptoms": [
        "Swelling (edema) in feet/ankles",
        "Brittle / Discoloured Hair"
        ],
        "observations": "asdf"
    },
    "photos": [
        {
        "type": "frontView",
        "description": "Patient at 2 - 3 feet facing camera, full body visible, arms at sides",
        "url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131874/nouritrack/screenings/hqxcwb4ehgkrvks6wfxd.jpg"
        },
        {
        "type": "sideProfile",
        "description": "Patient's left side view showing posture, full body visible, arms at sides",
        "url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qymlm1wb3y9flce5uslg.jpg"
        },
        {
        "type": "faceCloseUp",
        "description": "Clear view of facial features with focus on eyes and mouth",
        "url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/vma6wutydp0a3cy4gpmc.jpg"
        },
        {
        "type": "arm",
        "description": "Patient's left upper arm showing muscle mass",
        "url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qycyjl3rlvaz6nfynmwq.jpg"
        }
    ],
    "screeningDate": "2025-11-26T04:37:49.257Z"
    }
    """

    """
    frontView -> leg
    sideProfile -> side
    faceCloseUp -> head
    arm -> muac
    """

    json_schema_example = {
       "screening_id": "69268423ed8293d3b66aeeda", 
        "body_parts": [
            {
                "body_part": "muac",
                "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qycyjl3rlvaz6nfynmwq.jpg",
            },
            {
                "body_part": "leg",
                "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131874/nouritrack/screenings/hqxcwb4ehgkrvks6wfxd.jpg",
            },
            {
                "body_part": "head",
                "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/vma6wutydp0a3cy4gpmc.jpg",
            },
            {
                "body_part": "side",
                "image_url": "https://res.cloudinary.com/da3f8b0ym/image/upload/v1764131873/nouritrack/screenings/qymlm1wb3y9flce5uslg.jpg",
            },
        ],
        "vital_measurements": {
            "muac": 12.0,
            "height": 85.0,
            "weight": 10.0,
            "age": 36.0,
            "whz": -2.0,
            "waz": -3.0,
            "haz": -2.0,
        },
        "clinical_findings": {
            "symptoms": [
                "Swelling (edema) in feet/ankles",
                "Brittle / Discoloured Hair",
            ],
            "observations": "Additional clinical observations here",
        },
    }

    import requests
    import json

    print(json.dumps(json_schema_example, indent=2))
    response = requests.post("http://localhost:8000/predict", json=json_schema_example)
    print("Status Code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=2))
