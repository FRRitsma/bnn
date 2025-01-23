from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    root_path: Path = Path(__file__).parent
    data_path: Path = root_path / "data"
    models_path: Path = data_path / "models"


settings = Settings()
