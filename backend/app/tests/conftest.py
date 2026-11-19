import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Configure isolated SQLite DB and temp storage BEFORE importing app
_tmp_dir = Path(tempfile.mkdtemp(prefix="medvision_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_dir / 'test_pacs.db'}"
os.environ["STORAGE_DIR"] = str(_tmp_dir / "storage")
os.environ["MODELS_DIR"] = str(_tmp_dir / "models")
os.environ["REPORTS_DIR"] = str(_tmp_dir / "reports")
os.environ["CXR_WEIGHTS_PATH"] = str(_tmp_dir / "models" / "densenet121_nih14.pth")
os.environ["SEG_WEIGHTS_PATH"] = str(_tmp_dir / "models" / "monai_unet3d_task06_lung.pth")
os.environ["REQUIRE_AUTH"] = "false"

from app.database.session import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
