import os
from dotenv import load_dotenv
from langchain.agents import create_agent

from schema.diagnosis import DiagnosticReport
from utils.prompt import SYSTEM_PROMPT

load_dotenv()
agent = create_agent(
    model="gpt-5-mini",
    system_prompt=SYSTEM_PROMPT,
    response_format=DiagnosticReport
)

contents = ''' 
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

'''
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": contents}]
    })
    print(result["structured_response"].model_dump_json(indent=2))
