from pathlib import Path
import numpy as np
import torch

from app.ml.classification.model import (
    NIH_CHESTXRAY14_LABELS,
    ChestXRayDenseNet121,
    ModelWeightsUnavailableError,
)
from app.ml.classification.predict import predict_chest_xray
from app.ml.segmentation.evaluate import compute_binary_segmentation_metrics
from app.ml.segmentation.model import build_monai_unet3d
from app.ml.segmentation.predict import predict_lung_ct_volume
from app.services.dicom_service import apply_window_and_photometric
from app.services.gradcam_service import generate_gradcam_artifacts
from app.services.preprocessing_service import (
    preprocess_2d_xray_from_uint8,
    preprocess_ct_volume_from_slices,
)


def test_monochrome1_inversion_and_windowing():
    arr = np.array([[0, 500], [1000, 2000]], dtype=np.float32)
    _, mono2 = apply_window_and_photometric(arr, window_center=1000, window_width=2000, photometric_interpretation="MONOCHROME2")
    _, mono1 = apply_window_and_photometric(arr, window_center=1000, window_width=2000, photometric_interpretation="MONOCHROME1")
    assert mono2[0, 0] == 0 and mono2[1, 1] == 255
    assert mono1[0, 0] == 255 and mono1[1, 1] == 0


def test_missing_checkpoint_raises_explicit_error(tmp_path: Path):
    tensor, _ = preprocess_2d_xray_from_uint8(np.zeros((128, 128), dtype=np.uint8))
    missing = tmp_path / "does_not_exist.pth"
    try:
        predict_chest_xray(tensor, missing, device_pref="cpu")
        assert False, "Expected ModelWeightsUnavailableError"
    except ModelWeightsUnavailableError:
        assert True


def test_densenet121_inference_and_real_gradcam_generation(tmp_path: Path):
    """
    Verifies DenseNet-121 forward inference AND real gradient-based Grad-CAM generation
    using a temporary saved checkpoint (tests code pipeline without claiming clinical accuracy).
    """
    model = ChestXRayDenseNet121(num_classes=len(NIH_CHESTXRAY14_LABELS), pretrained=False)
    ckpt_path = tmp_path / "temp_densenet121.pth"
    torch.save({"model_state_dict": model.state_dict(), "labels": NIH_CHESTXRAY14_LABELS}, ckpt_path)

    sample_img = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    tensor, orig_rgb = preprocess_2d_xray_from_uint8(sample_img)
    assert tuple(tensor.shape) == (1, 3, 224, 224)

    result, loaded_model, device = predict_chest_xray(tensor, ckpt_path, device_pref="cpu")
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["all_probabilities"]) == 14

    heatmap_path = tmp_path / "gradcam.png"
    blended_path = tmp_path / "blended.png"
    cam_meta = generate_gradcam_artifacts(
        model=loaded_model,
        input_tensor=tensor,
        original_rgb=orig_rgb,
        target_class_index=result["top_class_index"],
        device=device,
        heatmap_output_path=heatmap_path,
        blended_output_path=blended_path,
    )
    assert heatmap_path.exists()
    assert blended_path.exists()
    assert "causation" in cam_meta["disclaimer"].lower()


def test_monai_3d_unet_segmentation_inference(tmp_path: Path):
    model = build_monai_unet3d()
    ckpt_path = tmp_path / "temp_unet3d.pth"
    torch.save({"model_state_dict": model.state_dict(), "roi_size": [32, 32, 32]}, ckpt_path)

    hu_slices = [np.full((32, 32), -600.0, dtype=np.float32) for _ in range(32)]
    vol_tensor, norm_vol = preprocess_ct_volume_from_slices(hu_slices)
    assert tuple(vol_tensor.shape) == (1, 1, 32, 32, 32)
    assert float(norm_vol.min()) >= 0.0 and float(norm_vol.max()) <= 1.0

    binary_mask, prob_map, summary = predict_lung_ct_volume(
        vol_tensor, ckpt_path, device_pref="cpu"
    )
    assert binary_mask.shape == (32, 32, 32)
    assert prob_map.shape == (32, 32, 32)
    assert "estimated_volume_mm3" in summary


def test_segmentation_metrics_calculation():
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    metrics = compute_binary_segmentation_metrics(pred, truth)
    assert abs(metrics["dice"] - 0.5) < 1e-4
    assert abs(metrics["sensitivity"] - 0.5) < 1e-4
    assert abs(metrics["specificity"] - 0.5) < 1e-4
