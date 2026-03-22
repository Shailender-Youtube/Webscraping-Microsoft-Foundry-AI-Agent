from typing import List, Optional
from pydantic import BaseModel, Field


class SalesBrief(BaseModel):
    company_name: Optional[str] = Field(default="", description="Full company name")
    what_they_do: Optional[str] = Field(default="", description="Core product or service in 2-3 sentences")
    industry: Optional[str] = Field(default="", description="Primary industry or sector")
    company_size_signals: Optional[List[str]] = Field(default_factory=list, description="Employee count hints, office locations, scale signals")
    recent_news: Optional[List[str]] = Field(default_factory=list, description="Last 3-5 notable announcements, acquisitions, or events")
    industries_served: Optional[List[str]] = Field(default_factory=list, description="Verticals or customer segments they target")
    spend_signals: Optional[List[str]] = Field(default_factory=list, description="Budget areas, cloud/tech investments, hiring signals, partnerships")
    key_people: Optional[List[str]] = Field(default_factory=list, description="Named executives especially CTO, CIO, VP Engineering, IT leads")
    tech_stack_mentions: Optional[List[str]] = Field(default_factory=list, description="Any cloud, platform, or software technology mentioned on the site")
    current_challenges: Optional[List[str]] = Field(default_factory=list, description="Pain points or problems they publicly acknowledge")
    talking_points: Optional[List[str]] = Field(default_factory=list, description="3-5 angles a sales rep should lead with in the first meeting")
    open_questions: Optional[List[str]] = Field(default_factory=list, description="Smart discovery questions the rep should ask the prospect")
    azure_relevance: Optional[str] = Field(default="", description="1-2 sentences on where Microsoft/Azure solutions could be relevant")
