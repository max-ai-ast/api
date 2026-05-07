from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..lib.diversify import mmr_rerank
from ..models import CandidatePost
from ..security import verify_api_key

router = APIRouter(tags=["diversify"], dependencies=[Depends(verify_api_key)])


class DiversifyRequest(BaseModel):
    candidates: list[CandidatePost]


class DiversifyResponse(BaseModel):
    candidates: list[CandidatePost]


@router.post("/diversify", response_model=DiversifyResponse)
async def diversify(payload: DiversifyRequest) -> DiversifyResponse:
    return DiversifyResponse(candidates=mmr_rerank(payload.candidates))
