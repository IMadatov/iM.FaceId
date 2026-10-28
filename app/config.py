from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "faces"
    # Override via .env (see .env.example). Do not commit real passwords.
    database_url: str = "postgresql+psycopg://imfaceid@localhost:5432/imfaceid"
    face_match_threshold: float = 0.40
    face_search_min_score: float = 0.40
    face_search_default_limit: int = 5
    face_search_max_limit: int = 20
    face_groups_min_size: int = 2
    face_groups_max_faces: int = 5000  # legacy; unused by ANN groups path
    face_groups_ann_top_k: int = 20
    face_groups_keep_runs: int = 2
    face_list_default_limit: int = 100
    face_list_max_limit: int = 1000
    model_dir: str = "./models"
    liveness_model_path: str = "./models/liveness/minifasnet_v2.onnx"
    liveness_threshold: float = 0.50
    liveness_crop_scale: float = 2.7
    liveness_max_frames: int = 5
    liveness_min_distance: float = 0.05  # NEEDS TUNING: min max-pairwise cosine distance
    # Motion-parallax check. Synthetic 25mm-nose faces score 0.12 at 15 deg of
    # yaw and 0.16 at 20; a flat photo scores 0.00 at every angle. Set to 0 to
    # disable both this and the motion gate below.
    liveness_min_planarity: float = 0.06  # NEEDS TUNING on real captures
    # Minimum observed turn before the planarity score carries information.
    # Scale is log face-aspect: ~0.015 at 10 deg of yaw, 0.035 at 15, 0.062 at
    # 20, independent of camera distance. 0.025 is ~13 deg, where a real face
    # scores ~0.10 against the 0.06 planarity floor; a 5 deg wobble reaches only
    # 0.009. The ceiling is set by left/right alternations, which fall back on a
    # distance-decaying cue and reach just 0.030 at 1.5m.
    liveness_min_view_change: float = 0.025
    # Face quality (liveness=true path only); relax to disable a check.
    face_blur_min_variance: float = 50.0
    face_brightness_min: float = 40.0
    face_brightness_max: float = 220.0
    face_min_side_px: float = 80.0
    max_image_side: int = 640
    max_upload_bytes: int = 5_000_000
    embedding_dim: int = 512
    face_device: str = "auto"


def get_settings() -> Settings:
    return Settings()
