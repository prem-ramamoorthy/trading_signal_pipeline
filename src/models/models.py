from pydantic import BaseModel, Field, model_validator

MIN_CONTEXT_ROWS = 120

class Candle(BaseModel):
    time:   str
    open:   float = Field(..., gt=0)
    high:   float = Field(..., gt=0)
    low:    float = Field(..., gt=0)
    close:  float = Field(..., gt=0)
    Volume: float = Field(..., ge=0)

    @model_validator(mode="after")
    def ohlc_sanity(self) -> "Candle":
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")
        if self.low  > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")
        return self

class CandleList(BaseModel):
    candles: list[Candle] = Field(
        ...,
        min_length=MIN_CONTEXT_ROWS,
        description=f"At least {MIN_CONTEXT_ROWS} raw OHLCV candles (newest last).",
    )

class PredictionOut(BaseModel):
    colour:      str   = Field(..., description="GREEN or RED")
    green_prob:  float = Field(..., description="P(next candle is green)")
    red_prob:    float = Field(..., description="P(next candle is red)")
    confidence:  float = Field(..., description="Probability of the predicted class")
    signal:      int   = Field(..., description="1 = GREEN, 0 = RED")
    last_candle: str   = Field(..., description="Timestamp of the most recent input candle")
    latency_ms:  float = Field(..., description="End-to-end processing time in ms")

class BatchPredictionOut(BaseModel):
    n_predictions: int
    predictions:   list[dict]
    latency_ms:    float

class LivePredictionOut(BaseModel):
    next_candle_colour: str   = Field(..., description="Predicted colour of the NEXT candle: GREEN or RED")
    signal:             int   = Field(..., description="1 = GREEN, 0 = RED")
    green_prob:         float = Field(..., description="P(next candle is green)")
    red_prob:           float = Field(..., description="P(next candle is red)")
    confidence:         float = Field(..., description="Probability of the predicted class")
    current_candle: dict = Field(..., description="OHLCV of the most recent completed candle")
    current_colour: str  = Field(..., description="Colour of the current candle: GREEN or RED")
    ticker:        str   = Field(..., description="Yahoo Finance ticker used")
    interval:      str   = Field(..., description="Bar interval fetched")
    n_candles_used: int  = Field(..., description="Number of candles fed to the pipeline")
    fetched_at:    str   = Field(..., description="UTC timestamp when data was fetched")
    latency_ms:    float = Field(..., description="End-to-end latency in ms")
    
class DataDownloadOut(BaseModel):
    ticker:        str   = Field(..., description="Yahoo Finance ticker downloaded")
    interval:      str   = Field(..., description="Bar interval")
    days:          int   = Field(..., description="Lookback window in days")
    n_bars:        int   = Field(..., description="Total candles saved after dedup + cleaning")
    date_from:     str   = Field(..., description="Earliest candle timestamp in the file")
    date_to:       str   = Field(..., description="Latest candle timestamp in the file")
    csv_path:      str   = Field(..., description="Absolute path of the saved CSV file")
    file_size_kb:  float = Field(..., description="Size of the saved CSV in KB")
    chunks_fetched: int  = Field(..., description="Number of yfinance requests made")
    latency_ms:    float = Field(..., description="Total download + save time in ms")