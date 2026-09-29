from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def _test_image() -> bytes:
	"""Load the fixture image used by the multipart upload tests."""
	return (Path(__file__).parent / "fixtures" / "sample.webp").read_bytes()


def test_health_reports_loaded_model() -> None:
	with TestClient(app) as client:
		response = client.get("/health")

	assert response.status_code == 200
	assert response.json()["model_loaded"] is True


def test_predict_returns_top_k_predictions() -> None:
	top_k = 5
	with TestClient(app) as client:
		response = client.post(
			"/predict",
			files={"file": ("sample.webp", _test_image(), "image/webp")},
			data={"top_k": str(top_k)},
		)

	assert response.status_code == 200
	body = response.json()
	assert "predictions" in body
	assert len(body["predictions"]) == top_k
	for prediction in body["predictions"]:
		assert 0 <= prediction["confidence"] <= 1


def test_predict_rejects_invalid_top_k() -> None:
	with TestClient(app) as client:
		for top_k in (0, 11):
			response = client.post(
				"/predict",
				files={"file": ("sample.webp", _test_image(), "image/webp")},
				data={"top_k": str(top_k)},
			)
			assert response.status_code == 400


def test_predict_rejects_non_image_file() -> None:
	with TestClient(app) as client:
		response = client.post(
			"/predict",
			files={"file": ("not-an-image.txt", b"not an image", "text/plain")},
			data={"top_k": "5"},
		)

	assert response.status_code == 415
