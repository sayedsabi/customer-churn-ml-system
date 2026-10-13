
from enum import Enum

from pydantic import BaseModel, Field


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Gender(str, Enum):
    female = "Female"
    male = "Male"


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceType(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    none = "No"


class PaymentMethodType(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class MultipleLinesType(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


class InternetOptionType(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


class CustomerInput(BaseModel):
    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: YesNo
    Dependents: YesNo

    tenure: int = Field(ge=0, le=100)

    PhoneService: YesNo
    MultipleLines: MultipleLinesType

    InternetService: InternetServiceType
    OnlineSecurity: InternetOptionType
    OnlineBackup: InternetOptionType
    DeviceProtection: InternetOptionType
    TechSupport: InternetOptionType
    StreamingTV: InternetOptionType
    StreamingMovies: InternetOptionType

    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodType

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

class PredictionResponse(BaseModel):
    churn_prediction: int = Field(ge=0, le=1)
    churn_probability: float = Field(ge=0, le=1)
    interpretation: str
