def test_environment_registry_import_is_optional_dependency_safe() -> None:
    from tracerigor.env import environment_status

    status = environment_status()
    assert "blackjack" in status
    assert all(detail["status"] in {"available", "unavailable"} for detail in status.values())
