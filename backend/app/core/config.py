from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MedVision AI — Medical Imaging & Radiologist Review API"
    app_env: str = "development"
    database_url: str = "sqlite:///./medvision_local.db"

    orthanc_url: str = "http://orthanc:8042"
    orthanc_username: str = "orthanc"
    orthanc_password: str = "orthanc"

    storage_dir: Path = Path("./storage")
    models_dir: Path = Path("./models")
    reports_dir: Path = Path("./reports")
    max_upload_mb: int = 50

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    require_auth: bool = False
    api_auth_token: str = ""

    cxr_weights_path: Path = Path("./models/densenet121_nih14.pth")
    seg_weights_path: Path = Path("./models/monai_unet3d_task06_lung.pth")
    device: str = "auto"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        path = self.storage_dir / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def artifacts_dir(self) -> Path:
        path = self.storage_dir / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
