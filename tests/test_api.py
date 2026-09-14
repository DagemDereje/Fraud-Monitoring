from src.app import TransactionRequest, risk_level, risk_reasons


def test_risk_reasons_and_level():
    assert risk_level(0.9, True) == "HIGH"
    assert risk_level(0.5, False) == "MEDIUM"
    assert risk_level(0.1, False) == "LOW"


def test_high_risk_reasons():
    tx = TransactionRequest(
        transaction_id="TXN_TEST",
        timestamp="2026-09-03T10:00:00+00:00",
        account_id="ACC_00001",
        amount=8500,
        merchant_category="jewelry",
        location_mismatch=1,
        device_velocity=6,
    )
    reasons = risk_reasons(tx)
    assert any("$5,000" in reason for reason in reasons)
    assert any("location" in reason.lower() for reason in reasons)
    assert any("velocity" in reason.lower() for reason in reasons)
    assert any("combined" in reason.lower() for reason in reasons)
