from src.data_loader import DataLoader


def test_wti_alias_candidates_include_proxy():
    loader = DataLoader("WTI", "2024-01-01", "2024-02-01")
    candidates = loader._candidate_tickers()
    assert "USO" in candidates
    assert candidates[0] in {"CL1!", "CL=F", "USO"}


def test_regular_symbol_candidates_identity():
    loader = DataLoader("NVDA", "2024-01-01", "2024-02-01")
    assert loader._candidate_tickers() == ["NVDA"]
