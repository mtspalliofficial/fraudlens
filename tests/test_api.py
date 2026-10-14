import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fraudlens import api  # noqa: E402
from fraudlens.config import RAW_COLUMNS  # noqa: E402
from fraudlens.service import FraudService  # noqa: E402


@pytest.fixture
def client(trained):
    art, df, _ = trained
    with TestClient(api.app) as c:
        api.state["service"] = FraudService(art)
        yield c, df


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_score(client):
    c, df = client
    body = df[RAW_COLUMNS].iloc[0].to_dict()
    r = c.post("/score", json=body)
    assert r.status_code == 200
    assert set(r.json()) >= {"fraud_probability", "decision", "top_reasons"}


def test_rejects_negative_amount(client):
    c, df = client
    body = df[RAW_COLUMNS].iloc[0].to_dict() | {"Amount": -5}
    assert c.post("/score", json=body).status_code == 422
