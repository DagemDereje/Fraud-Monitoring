from src.generator import TransactionGenerator


def test_transaction_schema():
    tx = TransactionGenerator(seed=1).generate_transaction()
    expected = {
        "transaction_id", "timestamp", "account_id", "amount",
        "merchant_category", "location_mismatch", "device_velocity", "is_fraud",
    }
    assert expected.issubset(tx.keys())
    assert tx["amount"] > 0
    assert tx["location_mismatch"] in (0, 1)
    assert tx["device_velocity"] >= 0


def test_historical_dataset_has_rare_fraud():
    df = TransactionGenerator(seed=42).generate_historical_data(10_000)
    rate = df["is_fraud"].mean()
    assert 0.005 <= rate <= 0.01
    assert (df["amount"] > 5000).any()
    assert ((df["location_mismatch"] == 1) & (df["device_velocity"] > 3)).any()


def test_controlled_scenarios_exercise_expected_signals():
    generator = TransactionGenerator(seed=7)

    normal = generator.generate_transaction(scenario="normal")
    high_value = generator.generate_transaction(scenario="high_value")
    location = generator.generate_transaction(scenario="location_mismatch")
    velocity = generator.generate_transaction(scenario="high_velocity")
    combined = generator.generate_transaction(scenario="combined")

    assert normal["amount"] <= 300
    assert normal["location_mismatch"] == 0
    assert normal["device_velocity"] <= 2

    assert high_value["amount"] > 5000
    assert location["location_mismatch"] == 1
    assert velocity["device_velocity"] > 3
    assert combined["amount"] > 5000
    assert combined["location_mismatch"] == 1
    assert combined["device_velocity"] > 3
