from pydantic import BaseModel, Field


class GlobalSyntheaConfig(BaseModel):
    install_directory: str


class GlobalOpenAPIConfig(BaseModel):
    api_key: str = Field(repr=False)
    batch_size: int
    delay_on_error: int
    backoff_on_error: int
    max_delay_on_error: int
    retry_attempts: int


class ComprehendMedicalConfig(BaseModel):
    profile: str
    service: str


class CacheConfig(BaseModel):
    file_name: str


class JuliaConfig(BaseModel):
    path_to_julia: str


class GlobalConfig(BaseModel):
    synthea: GlobalSyntheaConfig
    openai: GlobalOpenAPIConfig
    comprehendmedical: ComprehendMedicalConfig
    cache: CacheConfig
    pcm: JuliaConfig
